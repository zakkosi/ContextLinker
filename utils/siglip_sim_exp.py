import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

# SigLIP 프로세서 import (processors 폴더에서)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors.siglip2_processor import SigLIP2Processor

def run_similarity_experiment(image_pairs: List[Tuple[str, str]], 
                            experiment_name: str) -> Dict[str, any]:
    """
    SigLIP 유사도 실험 실행
    
    Args:
        image_pairs: (GT이미지, 비교이미지) 튜플 리스트
        experiment_name: 실험 이름
        
    Returns:
        dict: 실험 결과
    """
    print(f"\n=== {experiment_name} 시작 ===")
    
    processor = SigLIP2Processor()
    
    results = []
    total_time = 0
    
    for i, (gt_image, compare_image) in enumerate(image_pairs, 1):
        print(f"[{i}/{len(image_pairs)}] 처리 중:")
        print(f"  GT: {Path(gt_image).name}")
        print(f"  비교: {Path(compare_image).name}")
        
        try:
            # SigLIP2의 내장 유사도 계산 사용
            result = processor.compute_similarity(gt_image, "")
            
            # 두 이미지 간 유사도를 위해 직접 계산
            gt_features, gt_time = processor.encode_image(gt_image)
            compare_features, compare_time = processor.encode_image(compare_image)
            
            # PyTorch의 코사인 유사도 함수 사용
            import torch.nn.functional as F
            similarity = F.cosine_similarity(gt_features, compare_features).item()
            
            processing_time = gt_time + compare_time
            total_time += processing_time
            
            result = {
                'pair_id': i,
                'gt_image': Path(gt_image).name,
                'compare_image': Path(compare_image).name,
                'similarity_score': similarity,
                'processing_time': processing_time
            }
            
            results.append(result)
            
            print(f"  유사도: {similarity:.4f}")
            print(f"  처리시간: {processing_time:.3f}초")
            
        except Exception as e:
            print(f"  오류: {e}")
            results.append({
                'pair_id': i,
                'gt_image': Path(gt_image).name,
                'compare_image': Path(compare_image).name,
                'error': str(e)
            })
    
    # 통계 계산
    valid_results = [r for r in results if 'similarity_score' in r]
    
    if valid_results:
        similarities = [r['similarity_score'] for r in valid_results]
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        min_similarity = min(similarities)
    else:
        avg_similarity = max_similarity = min_similarity = 0
    
    experiment_result = {
        'experiment_name': experiment_name,
        'total_pairs': len(image_pairs),
        'successful_pairs': len(valid_results),
        'failed_pairs': len(image_pairs) - len(valid_results),
        'avg_similarity': avg_similarity,
        'max_similarity': max_similarity,
        'min_similarity': min_similarity,
        'total_processing_time': total_time,
        'results': results
    }
    
    print(f"\n{experiment_name} 결과:")
    print(f"  성공: {len(valid_results)}/{len(image_pairs)}")
    print(f"  평균 유사도: {avg_similarity:.4f}")
    print(f"  최대 유사도: {max_similarity:.4f}")
    print(f"  최소 유사도: {min_similarity:.4f}")
    
    return experiment_result

def find_image_pairs(base_dir: str = ".") -> Dict[str, List[Tuple[str, str]]]:
    """
    이미지 파일들을 찾아서 GT-MN, GT-SCH 페어 구성
    
    Args:
        base_dir: 이미지들이 있는 디렉토리
        
    Returns:
        dict: 실험별 이미지 페어 딕셔너리
    """
    base_path = Path(base_dir)
    
    # 모든 이미지 파일 찾기
    image_files = list(base_path.glob("*.jpg")) + list(base_path.glob("*.png"))
    
    # 파일명 분석
    gt_images = [f for f in image_files if f.stem.endswith('_GT')]
    mn_images = [f for f in image_files if f.stem.endswith('_MN')]
    sch_images = [f for f in image_files if f.stem.endswith('_SCH')]
    
    print(f"발견된 이미지:")
    print(f"  GT 이미지: {len(gt_images)}개")
    for gt in gt_images:
        print(f"    {gt.name}")
    print(f"  MN 이미지: {len(mn_images)}개")
    for mn in mn_images:
        print(f"    {mn.name}")
    print(f"  SCH 이미지: {len(sch_images)}개")
    for sch in sch_images:
        print(f"    {sch.name}")
    
    experiments = {}
    
    # GT vs MN 페어 만들기
    gt_mn_pairs = []
    for gt_img in gt_images:
        # sig_01_GT.jpg -> sig_01_MN.png 매칭 (확장자 다를 수 있음)
        base_name = gt_img.stem.replace('_GT', '')
        
        # MN 이미지 중에서 같은 base_name 찾기
        mn_match = None
        for mn_img in mn_images:
            if mn_img.stem.replace('_MN', '') == base_name:
                mn_match = mn_img
                break
        
        if mn_match:
            gt_mn_pairs.append((str(gt_img), str(mn_match)))

    experiments['GT_vs_MN'] = gt_mn_pairs
    
    # GT vs SCH 페어 만들기
    gt_sch_pairs = []
    for gt_img in gt_images:
        base_name = gt_img.stem.replace('_GT', '')
        
        # SCH 이미지 중에서 같은 base_name 찾기
        sch_match = None
        for sch_img in sch_images:
            if sch_img.stem.replace('_SCH', '') == base_name:
                sch_match = sch_img
                break
        
        if sch_match:
            gt_sch_pairs.append((str(gt_img), str(sch_match)))
    
    experiments['GT_vs_SCH'] = gt_sch_pairs
    
    print(f"\n구성된 실험:")
    print(f"  GT vs MN: {len(gt_mn_pairs)} 페어")
    print(f"  GT vs SCH: {len(gt_sch_pairs)} 페어")
    
    return experiments

def save_experiment_results(results: List[Dict], output_file: str = "siglip_experiment_results.json"):
    """실험 결과를 JSON 파일로 저장"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n실험 결과 저장: {output_file}")

def print_summary(all_results: List[Dict]):
    """전체 실험 결과 요약 출력"""
    print(f"\n{'='*50}")
    print(f"전체 실험 결과 요약")
    print(f"{'='*50}")
    
    for result in all_results:
        name = result['experiment_name']
        avg_sim = result['avg_similarity']
        success_rate = result['successful_pairs'] / result['total_pairs'] * 100
        
        print(f"{name:15} | 평균 유사도: {avg_sim:6.4f} | 성공률: {success_rate:5.1f}%")
    
    # GT vs MN과 GT vs SCH 비교
    gt_mn_results = [r for r in all_results if 'MN' in r['experiment_name']]
    gt_sch_results = [r for r in all_results if 'SCH' in r['experiment_name']]
    
    if gt_mn_results and gt_sch_results:
        mn_avg = sum(r['avg_similarity'] for r in gt_mn_results) / len(gt_mn_results)
        sch_avg = sum(r['avg_similarity'] for r in gt_sch_results) / len(gt_sch_results)
        
        print(f"\n전체 평균:")
        print(f"GT vs Manual:     {mn_avg:.4f}")
        print(f"GT vs Schrodinger: {sch_avg:.4f}")
        print(f"차이:            {abs(sch_avg - mn_avg):+.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python siglip_similarity_experiment.py <이미지디렉토리>")
        print("예시: python siglip_similarity_experiment.py ../data/experiment_images")
        exit()
    
    image_dir = sys.argv[1]
    
    try:
        # 이미지 페어 찾기
        experiments = find_image_pairs(image_dir)
        
        if not experiments['GT_vs_MN'] and not experiments['GT_vs_SCH']:
            print("매칭되는 이미지 페어를 찾을 수 없습니다.")
            exit()
        
        all_results = []
        start_time = time.time()
        
        # GT vs MN 실험
        if experiments['GT_vs_MN']:
            result_mn = run_similarity_experiment(
                experiments['GT_vs_MN'], 
                "GT vs Manual"
            )
            all_results.append(result_mn)
        
        # GT vs SCH 실험
        if experiments['GT_vs_SCH']:
            result_sch = run_similarity_experiment(
                experiments['GT_vs_SCH'], 
                "GT vs Schrodinger"
            )
            all_results.append(result_sch)
        
        total_experiment_time = time.time() - start_time
        
        # 결과 저장
        save_experiment_results(all_results)
        
        # 요약 출력
        print_summary(all_results)
        
        print(f"\n총 실험 시간: {total_experiment_time:.2f}초")
        
    except Exception as e:
        print(f"실험 실행 중 오류: {e}")