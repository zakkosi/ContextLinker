import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import shutil
from datetime import datetime

# 프로세서들 import
from processors.sam2_processor import SAM2Segmenter
from processors.colpali_processor import ColPaliProcessor
from processors.llm_processor import LLMProcessor
from processors.schrodinger_processor import SchrodingerBridgeProcessor

class BMWManualAssistant:
    def __init__(self, pdf_images_folder: str = "output/PDF_Images"):
        """BMW 매뉴얼 어시스턴트 초기화"""
        print("BMW Manual Assistant 초기화 중...")
        
        # 프로세서들 초기화
        self.sam2_processor = SAM2Segmenter()
        self.colpali_processor = ColPaliProcessor()
        self.llm_processor = LLMProcessor()
        self.schrodinger_processor = SchrodingerBridgeProcessor()
        
        # ColPali 데이터베이스 구축
        self.pdf_images_folder = pdf_images_folder
        print(f"ColPali 데이터베이스 초기화 중: {self.pdf_images_folder}")
        
        try:
            num_images, build_time = self.colpali_processor.build_database_from_pdf_images(
                self.pdf_images_folder, 
                batch_size=8
            )
            print(f"ColPali 데이터베이스 구축 완료: {num_images}개 이미지, {build_time:.2f}초")
            
        except Exception as e:
            print(f"ColPali 데이터베이스 구축 실패: {e}")
            raise
        
        self.schrodinger_mode = "real_to_illust"
        print("BMW Manual Assistant 초기화 완료")
    
    def _apply_mask_to_schrodinger(self, schrodinger_image_path: str, sam2_result: Dict[str, Any]) -> str:
        """
        슈뢰딩거 이미지에 SAM2 마스크를 적용하여 마스크된 이미지 생성
        
        Args:
            schrodinger_image_path: 슈뢰딩거 처리된 이미지 경로
            sam2_result: SAM2 처리 결과 (마스크 포함)
            
        Returns:
            str: 마스크된 슈뢰딩거 이미지 경로
        """
        from PIL import Image
        import numpy as np
        
        try:
            # 슈뢰딩거 이미지 로드
            schrodinger_img = Image.open(schrodinger_image_path).convert("RGBA")
            
            # SAM2 마스크 추출 (sam2_result에서 마스크 정보 가져오기)
            if 'mask' in sam2_result:
                mask = sam2_result['mask']
                if isinstance(mask, np.ndarray):
                    # numpy 배열을 PIL Image로 변환
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                else:
                    # 이미 PIL Image인 경우
                    mask_img = mask
            else:
                # 마스크가 없으면 전체 이미지 사용
                mask_img = Image.new('L', schrodinger_img.size, 255)
            
            # 마스크 크기를 슈뢰딩거 이미지와 맞춤
            mask_img = mask_img.resize(schrodinger_img.size, Image.Resampling.LANCZOS)
            
            # 슈뢰딩거 이미지에 마스크 적용
            schrodinger_array = np.array(schrodinger_img)
            mask_array = np.array(mask_img)
            
            # 마스크 영역만 유지, 나머지는 투명하게
            masked_array = schrodinger_array.copy()
            masked_array[:, :, 3] = mask_array  # 알파 채널에 마스크 적용
            
            # 최종 이미지 생성
            final_img = Image.fromarray(masked_array, 'RGBA')
            
            # 결과 저장
            output_path = f"temp_masked_schrodinger_{int(time.time())}.png"
            final_img.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"마스크 적용 실패: {e}")
            # 실패시 원본 슈뢰딩거 이미지 반환
            return schrodinger_image_path
    
    def process_request(self, request_data: Dict[str, Any], experiment_dir: str = None, test_id: int = 1) -> Dict[str, Any]:
        """
        Unity 요청 처리 - Pipeline v2
        
        Args:
            request_data: {
                "stt_text": str,
                "image_path": str,
                "mode": int (0 or 1),
                "bbox": List[float] (optional, if mode=1),                
            }
            
        Returns:
            Dict: 처리 결과
        """
        start_time = time.time()
        
        stt_text = request_data.get("stt_text", "")
        image_path = request_data.get("image_path", "")
        mode = request_data.get("mode", 0)
        bbox = request_data.get("bbox", None)
        schrodinger_mode = self.schrodinger_mode
        
        print(f"\n=== BMW Manual Assistant 요청 처리 (Pipeline v2) ===")
        print(f"STT: {stt_text}")
        print(f"이미지: {image_path}")
        print(f"모드: {mode}")
        print(f"BBox: {bbox}")
        print(f"슈뢰딩거 모드: {schrodinger_mode} (서버 고정값)")
        
        try:
            # 1. 슈뢰딩거 브릿지 처리 (원본 이미지 전체)
            print(f"\n1. 슈뢰딩거 브릿지 ({schrodinger_mode}) - 원본 이미지...")
            schrodinger_start = time.time()
            if schrodinger_mode == "real_to_illust":
                schrodinger_full_path = self.schrodinger_processor.real_to_illustration(image_path)
            elif schrodinger_mode == "illust_to_real":
                schrodinger_full_path = self.schrodinger_processor.illustration_to_real(image_path)
            else:
                raise ValueError(f"잘못된 슈뢰딩거 모드: {schrodinger_mode}")
            schrodinger_time = time.time() - schrodinger_start
            print(f"슈뢰딩거 전체 이미지 완료: {schrodinger_full_path}")
            
            # 2. SAM2로 마스크 생성
            print("\n2. SAM2 마스크 생성...")
            sam2_start = time.time()
            sam2_result = self.sam2_processor.process_image(image_path, mode, bbox)
            sam2_time = time.time() - sam2_start
            print(f"SAM2 처리 완료 (점수: {sam2_result['score']:.3f})")
            
            # 3. 마스크와 슈뢰딩거 이미지 합성
            print("\n3. 마스크된 슈뢰딩거 이미지 생성...")
            mask_start = time.time()
            processed_path = self._apply_mask_to_schrodinger(schrodinger_full_path, sam2_result)
            mask_time = time.time() - mask_start
            print(f"마스크된 슈뢰딩거 이미지 완료: {processed_path}")
            
            # 4. 이미지-이미지 유사도로 매뉴얼 검색
            print("\n4. 이미지 유사도로 매뉴얼 검색...")
            colpali_start = time.time()
            similar_pages = self.colpali_processor.search_by_image(processed_path, k=5)
            colpali_time = time.time() - colpali_start
            
            # 검색 결과 로깅
            print(f"검색 완료: Top-{len(similar_pages)} 결과")
            for i, result in enumerate(similar_pages, 1):
                print(f"  {i}. {result['image_name']} (유사도: {result['similarity_score']:.2f})")
            
            # 5. GPT-4o로 최종 답변 생성
            print("\n5. GPT-4o 최종 답변 생성...")
            llm_start = time.time()
            llm_result = self.llm_processor.generate_bmw_manual_response(
                user_prompt=stt_text,
                manual_pages=similar_pages,
                segmented_part=processed_path  # 마스크된 슈뢰딩거 이미지 전달
            )
            llm_time = time.time() - llm_start
            total_time = time.time() - start_time
            
            # 결과 구성
            result = {
                'success': True,
                'stt_text': stt_text,
                'schrodinger_full_path': schrodinger_full_path,  # 전체 슈뢰딩거 이미지
                'processed_image_path': processed_path,          # 마스크된 슈뢰딩거 이미지
                'schrodinger_mode': schrodinger_mode,
                'top_manual_pages': similar_pages,
                'final_response': llm_result.get('response', 'Response generation failed'),
                'processing_time': total_time,
                'detailed_times': {
                    'schrodinger_time': schrodinger_time,
                    'sam2_time': sam2_time,
                    'mask_time': mask_time,
                    'colpali_time': colpali_time,
                    'llm_time': llm_time,
                    'total_time': total_time
                }
            }

            if experiment_dir and test_id:
                experiment_folder = self.save_experiment_results(
                    result, 
                    schrodinger_full_path,  # 전체 슈뢰딩거 이미지
                    processed_path,         # 마스크된 슈뢰딩거 이미지
                    similar_pages,
                    experiment_dir,  
                    test_id          
                )
                print(f"실험 결과 저장: {experiment_folder}")
            else:
                print("실험 결과 저장 건너뛰기 (experiment_dir 또는 test_id 없음)")   
            
            print(f"\n=== 처리 완료 (총 {total_time:.2f}초) ===")
            print(f"최종 답변: {llm_result.get('response', 'No response')[:100]}...")
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"처리 실패: {e}")
            return error_result
    
    def cleanup_temp_files(self):
        """임시 파일들 정리"""
        import glob
        temp_files = glob.glob("temp_segmented_*.png") + glob.glob("temp_masked_schrodinger_*.png")
        for file in temp_files:
            try:
                os.remove(file)
            except:
                pass

    def save_experiment_results(self, result, schrodinger_full_path, processed_path, similar_pages, experiment_dir, test_id):
        """실험 결과를 지정된 폴더에 저장"""
        # 테스트별 서브폴더 생성
        test_dir = f"{experiment_dir}/test_{test_id:02d}"
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(f"{test_dir}/retrieved_pages", exist_ok=True)
        
        try:
            shutil.copy(schrodinger_full_path, f"{test_dir}/schrodinger_full.png")
            os.remove(schrodinger_full_path)  # 복사 후 즉시 삭제
            print(f"삭제: {schrodinger_full_path}")
        except Exception as e:
            print(f"파일 삭제 실패: {schrodinger_full_path}, {e}")
        
        try:
            shutil.copy(processed_path, f"{test_dir}/masked_schrodinger.png") 
            os.remove(processed_path)  # 복사 후 즉시 삭제
            print(f"삭제: {processed_path}")
        except Exception as e:
            print(f"파일 삭제 실패: {processed_path}, {e}")
        
        # 검색된 매뉴얼 페이지들 복사 (이건 temp 파일이 아니라서 삭제 안함)
        for page in similar_pages:
            page_name = page['image_name']
            shutil.copy(page['image_path'], f"{test_dir}/retrieved_pages/{page_name}")
        
        # JSON 결과 저장
        with open(f"{test_dir}/result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return test_dir

def load_test_cases(json_file: str = "test_cases.json") -> List[Dict[str, Any]]:
    """테스트 케이스 JSON 파일 로드"""
    if not os.path.exists(json_file):
        print(f"테스트 케이스 파일을 찾을 수 없습니다: {json_file}")
        return []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_test_pipeline():
    """파이프라인 테스트 실행"""
    
    # 테스트 케이스 로드
    test_cases = load_test_cases("TEST/test_cases.json")
    
    if not test_cases:
        print("테스트 케이스가 없습니다. test_cases.json 파일을 확인하세요.")
        return
    
    try:
        timestamp = datetime.now().strftime("%H%M%S")
        experiment_dir = f"output/experiments/exp_test_cases_v2_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)

        # BMW 어시스턴트 초기화
        assistant = BMWManualAssistant()
        
        # 각 테스트 케이스 실행
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"테스트 케이스 {i}/{len(test_cases)} - Pipeline v2")
            print(f"{'='*60}")
            
            result = assistant.process_request(test_case, experiment_dir, i)
            result['test_case_id'] = i
            results.append(result)
            
            if result['success']:
                print(f"\n✅ 테스트 {i} 성공")
            else:
                print(f"\n❌ 테스트 {i} 실패: {result['error']}")
        
        # 결과 저장
        with open(f'{experiment_dir}/all_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n실험 결과 저장: {experiment_dir}")
        
        # 성공률 계산
        successful_tests = len([r for r in results if r['success']])
        success_rate = (successful_tests / len(results)) * 100
        print(f"테스트 성공률: {successful_tests}/{len(results)} ({success_rate:.1f}%)")
        
        # 임시 파일 정리
        assistant.cleanup_temp_files()
        
    except Exception as e:
        print(f"테스트 실행 실패: {e}")

if __name__ == "__main__":
    # 환경 확인
    print("환경 확인 중...")
    
    required_folders = [
        "processors",
        "output/PDF_Images",
        "models/checkpoints"
    ]
    
    missing_folders = []
    for folder in required_folders:
        if not Path(folder).exists():
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"필요한 폴더가 없습니다: {missing_folders}")
        print("폴더를 생성하거나 경로를 수정하세요.")
    else:
        print("환경 확인 완료")
        run_test_pipeline()