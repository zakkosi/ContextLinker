import os
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from processors.colpali_processor import ColPaliProcessor

def run_self_retrieval_test(pdf_images_folder: str = "output/PDF_Images", 
                           num_test_images: int = 10, 
                           random_seed: int = 42) -> Dict[str, Any]:
    """
    ColPali self-retrieval 성능 테스트
    각 이미지가 자기 자신을 top-1으로 찾는지 확인
    
    Args:
        pdf_images_folder: PDF 이미지 폴더 경로
        num_test_images: 테스트할 이미지 수
        random_seed: 랜덤 시드
        
    Returns:
        Dict: 테스트 결과
    """
    print("🚀 ColPali Self-Retrieval 성능 테스트 시작")
    print("=" * 50)
    
    # 랜덤 시드 설정
    random.seed(random_seed)
    
    try:
        # ColPali 프로세서 초기화 및 데이터베이스 구축
        print("ColPali 프로세서 초기화 중...")
        processor = ColPaliProcessor()
        
        num_images, build_time = processor.build_database_from_pdf_images(pdf_images_folder)
        
        if num_images == 0:
            print("❌ 테스트할 이미지가 없습니다.")
            return {'success': False, 'error': 'No images found'}
        
        print(f"데이터베이스 구축 완료: {num_images}개 이미지, {build_time:.2f}초")
        
        # 데이터베이스 정보 가져오기
        db_info = processor.get_database_info()
        image_paths = db_info['image_paths']
        
        # 테스트할 이미지 수 조정
        actual_test_count = min(num_test_images, num_images)
        test_indices = random.sample(range(num_images), actual_test_count)
        
        print(f"테스트 대상: 전체 {num_images}개 중 랜덤 선택 {actual_test_count}개")
        print("=" * 50)
        
        # 테스트 실행
        results = []
        correct_count = 0
        total_test_time = 0
        
        for i, test_idx in enumerate(test_indices, 1):
            test_image_path = image_paths[test_idx]
            test_image_name = Path(test_image_path).name
            
            print(f"Test {i:2d}/{actual_test_count}: {test_image_name:<30}", end=" -> ")
            
            # 자기 자신을 쿼리로 검색
            start_time = time.time()
            search_results = processor.search_by_image(test_image_path, k=5)
            search_time = time.time() - start_time
            total_test_time += search_time
            
            if search_results:
                top1_path = search_results[0]['image_path']
                top1_name = search_results[0]['image_name']
                similarity = search_results[0]['similarity_score']
                
                is_correct = (top1_path == test_image_path)
                if is_correct:
                    correct_count += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                
                # 자기 자신이 몇 순위에 있는지 확인
                self_rank = None
                for rank, result in enumerate(search_results, 1):
                    if result['image_path'] == test_image_path:
                        self_rank = rank
                        break
                
                result_data = {
                    'test_number': i,
                    'test_image_name': test_image_name,
                    'test_image_path': test_image_path,
                    'top1_image_name': top1_name,
                    'top1_image_path': top1_path,
                    'top1_similarity': similarity,
                    'is_correct': is_correct,
                    'self_rank': self_rank,
                    'search_time': search_time,
                    'top5_results': search_results
                }
                
                print(f"{status} (유사도: {similarity:.4f})")
                if not is_correct:
                    print(f"           자기 순위: {self_rank}위, Top-1: {top1_name}")
                
            else:
                result_data = {
                    'test_number': i,
                    'test_image_name': test_image_name,
                    'test_image_path': test_image_path,
                    'is_correct': False,
                    'error': 'Search failed',
                    'search_time': search_time
                }
                print("❌ 검색 실패")
            
            results.append(result_data)
        
        # 통계 계산
        accuracy = correct_count / actual_test_count
        avg_search_time = total_test_time / actual_test_count
        
        # 결과 요약 출력
        print("\n" + "=" * 50)
        print("🎯 테스트 결과 요약")
        print("=" * 50)
        print(f"총 테스트 수: {actual_test_count}")
        print(f"성공한 검색: {correct_count}")
        print(f"정확도: {accuracy:.2%}")
        print(f"평균 검색 시간: {avg_search_time:.4f}초")
        print(f"총 테스트 시간: {total_test_time:.3f}초")
        
        # 성능 평가
        if accuracy == 1.0:
            print("\n🎉 완벽한 성능! 모든 이미지가 자기 자신을 top-1으로 찾았습니다.")
        elif accuracy >= 0.9:
            print(f"\n✅ 우수한 성능! {accuracy:.1%} 정확도입니다.")
        elif accuracy >= 0.7:
            print(f"\n⚠️  보통 성능. {accuracy:.1%} 정확도입니다. 일부 개선이 필요할 수 있습니다.")
        else:
            print(f"\n❌ 낮은 성능. {accuracy:.1%} 정확도입니다. ColPali 구현을 다시 확인해보세요.")
        
        # 실패 케이스 상세 분석
        failed_cases = [r for r in results if not r.get('is_correct', False)]
        if failed_cases:
            print(f"\n❌ 실패한 케이스들 ({len(failed_cases)}개):")
            for case in failed_cases[:5]:  # 최대 5개만 표시
                if 'error' not in case:
                    print(f"  - {case['test_image_name']}: 자기 순위 {case['self_rank']}위")
                    print(f"    Top-1은 {case['top1_image_name']} (유사도: {case['top1_similarity']:.4f})")
        
        # 전체 결과 반환
        summary = {
            'success': True,
            'total_tests': actual_test_count,
            'correct_retrievals': correct_count,
            'accuracy': accuracy,
            'avg_search_time': avg_search_time,
            'total_test_time': total_test_time,
            'database_size': num_images,
            'test_settings': {
                'pdf_images_folder': pdf_images_folder,
                'num_test_images': num_test_images,
                'random_seed': random_seed
            },
            'detailed_results': results
        }
        
        return summary
        
    except Exception as e:
        print(f"테스트 실행 실패: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def analyze_failure_cases(test_results: Dict[str, Any]) -> None:
    """실패 케이스들을 상세 분석"""
    if not test_results.get('success', False):
        print("❌ 테스트가 실패했습니다.")
        return
    
    failed_cases = [r for r in test_results['detailed_results'] if not r.get('is_correct', False)]
    
    if not failed_cases:
        print("🎉 실패한 케이스가 없습니다!")
        return
    
    print(f"\n🔍 실패 케이스 상세 분석 ({len(failed_cases)}개)")
    print("=" * 70)
    
    for i, case in enumerate(failed_cases, 1):
        print(f"\n[실패 케이스 {i}]")
        print(f"테스트 이미지: {case['test_image_name']}")
        
        if 'error' in case:
            print(f"오류: {case['error']}")
            continue
            
        print(f"자기 순위: {case['self_rank']}")
        print(f"자기 유사도: {case.get('self_similarity', 'N/A')}")
        print(f"Top-5 결과:")
        
        for j, result in enumerate(case.get('top5_results', []), 1):
            marker = "👈 자기자신" if result['image_path'] == case['test_image_path'] else ""
            print(f"  {j}위: {result['image_name']:<25} "
                  f"(유사도: {result['similarity_score']:.4f}) {marker}")

def save_test_results(test_results: Dict[str, Any], output_file: str = None) -> str:
    """테스트 결과를 JSON 파일로 저장"""
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"colpali_performance_test_{timestamp}.json"
    
    import json
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n📄 테스트 결과 저장: {output_file}")
        return output_file
    except Exception as e:
        print(f"결과 저장 실패: {e}")
        return ""

def main():
    """메인 실행 함수"""
    print("ColPali 성능 테스트 도구")
    print("=" * 30)
    
    # 설정
    pdf_images_folder = "output/PDF_Images"
    num_test_images = 10
    random_seed = 42
    
    # 테스트 실행
    results = run_self_retrieval_test(
        pdf_images_folder=pdf_images_folder,
        num_test_images=num_test_images,
        random_seed=random_seed
    )
    
    if results['success']:
        # 상세 분석
        analyze_failure_cases(results)
        
        # 결과 저장
        save_test_results(results)
        
        print(f"\n📊 최종 정확도: {results['accuracy']:.2%}")
        
    else:
        print(f"테스트 실패: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()