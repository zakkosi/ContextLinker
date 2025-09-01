import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import torch
from PIL import Image
from tqdm import tqdm

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
        
        # PDF 이미지들의 임베딩을 GPU에 미리 로드
        self.pdf_images_folder = pdf_images_folder
        self.db_embeddings = []
        self.db_image_paths = []
        self._preload_pdf_embeddings()
        
        print("BMW Manual Assistant 초기화 완료")
    
    def _preload_pdf_embeddings(self):
        """PDF 이미지들을 GPU에 미리 임베딩하여 로드"""
        print(f"PDF 이미지 임베딩을 GPU에 로드 중: {self.pdf_images_folder}")
        
        pdf_path = Path(self.pdf_images_folder)
        if not pdf_path.exists():
            raise ValueError(f"PDF 이미지 폴더를 찾을 수 없습니다: {self.pdf_images_folder}")
        
        # 모든 PDF 이미지 파일 수집
        all_images = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for pdf_folder in pdf_path.iterdir():
            if pdf_folder.is_dir():
                for ext in supported_extensions:
                    all_images.extend(list(pdf_folder.glob(f"*{ext}")))
                    all_images.extend(list(pdf_folder.glob(f"*{ext.upper()}")))
        
        if not all_images:
            raise ValueError("PDF 이미지를 찾을 수 없습니다")
        
        print(f"총 {len(all_images)}개 PDF 이미지 발견")
        
        # 배치 단위로 임베딩 생성
        batch_size = 8
        self.db_embeddings = []
        self.db_image_paths = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_images), batch_size), desc="PDF 이미지 임베딩"):
                batch_paths = all_images[i:i+batch_size]
                batch_images = []
                valid_paths = []
                
                # 이미지 로드
                for img_path in batch_paths:
                    try:
                        pil_image = Image.open(img_path).convert("RGB")
                        batch_images.append(pil_image)
                        valid_paths.append(str(img_path))
                    except Exception as e:
                        print(f"이미지 로드 실패 {img_path}: {e}")
                
                if batch_images:
                    # ColPali로 임베딩 생성
                    inputs = self.colpali_processor.processor.process_images(images=batch_images).to(self.colpali_processor.device)
                    batch_embeddings = list(torch.unbind(self.colpali_processor.model(**inputs).cpu()))
                    
                    self.db_embeddings.extend(batch_embeddings)
                    self.db_image_paths.extend(valid_paths)
        
        print(f"GPU에 {len(self.db_embeddings)}개 이미지 임베딩 로드 완료")
    
    def search_similar_manual_pages(self, query_image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """쿼리 이미지와 유사한 매뉴얼 페이지들 검색"""
        if not self.db_embeddings:
            raise ValueError("PDF 임베딩이 로드되지 않았습니다")
        
        try:
            # 쿼리 이미지 임베딩
            query_image = Image.open(query_image_path).convert("RGB")
            
            with torch.no_grad():
                query_inputs = self.colpali_processor.processor.process_images(images=[query_image]).to(self.colpali_processor.device)
                query_embedding = list(torch.unbind(self.colpali_processor.model(**query_inputs).cpu()))
                
                # 유사도 계산
                scores = self.colpali_processor.processor.score(query_embedding, self.db_embeddings, device=self.colpali_processor.device)
                top_k = scores[0].topk(k)
                
                indices = top_k.indices.tolist()
                similarities = top_k.values.tolist()
            
            # 결과 구성
            results = []
            for idx, sim_score in zip(indices, similarities):
                if idx < len(self.db_image_paths):
                    image_path = self.db_image_paths[idx]
                    results.append({
                        'image_name': Path(image_path).name,
                        'image_path': image_path,
                        'similarity_score': sim_score,
                        'rank': len(results) + 1
                    })
            
            return results
            
        except Exception as e:
            print(f"매뉴얼 페이지 검색 실패: {e}")
            return []
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unity 요청 처리
        
        Args:
            request_data: {
                "stt_text": str,
                "image_path": str,
                "mode": int (0 or 1),
                "bbox": List[float] (optional, if mode=1),
                "schrodinger_mode": str ("real_to_illust" or "illust_to_real", default="real_to_illust")
            }
            
        Returns:
            Dict: 처리 결과
        """
        start_time = time.time()
        
        stt_text = request_data.get("stt_text", "")
        image_path = request_data.get("image_path", "")
        mode = request_data.get("mode", 0)
        bbox = request_data.get("bbox", None)
        schrodinger_mode = request_data.get("schrodinger_mode", "real_to_illust")
        
        print(f"\n=== BMW Manual Assistant 요청 처리 ===")
        print(f"STT: {stt_text}")
        print(f"이미지: {image_path}")
        print(f"모드: {mode}")
        print(f"BBox: {bbox}")
        print(f"슈뢰딩거 모드: {schrodinger_mode}")
        
        try:
            # 1. SAM2로 세그멘테이션
            print("\n1. SAM2 세그멘테이션 수행...")
            sam2_result = self.sam2_processor.process_image(image_path, mode, bbox)
            
            # 세그먼트된 이미지 저장
            segmented_path = f"temp_segmented_{int(time.time())}.png"
            self.sam2_processor.save_result(sam2_result, segmented_path)
            print(f"세그멘테이션 완료: {segmented_path} (점수: {sam2_result['score']:.3f})")
            
            # 2. 슈뢰딩거 브릿지 처리
            print(f"\n2. 슈뢰딩거 브릿지 ({schrodinger_mode})...")
            if schrodinger_mode == "real_to_illust":
                processed_path = self.schrodinger_processor.real_to_illustration(segmented_path)
            elif schrodinger_mode == "illust_to_real":
                processed_path = self.schrodinger_processor.illustration_to_real(segmented_path)
            else:
                raise ValueError(f"잘못된 슈뢰딩거 모드: {schrodinger_mode}")
            
            # 3. 이미지-이미지 유사도로 매뉴얼 검색
            print("\n3. 이미지 유사도로 매뉴얼 검색...")
            similar_pages = self.search_similar_manual_pages(processed_path, k=5)
            
            print(f"검색 완료: Top-{len(similar_pages)} 결과")
            for i, result in enumerate(similar_pages, 1):
                print(f"  {i}. {result['image_name']} (유사도: {result['similarity_score']:.2f})")
            
            # 4. GPT-4o로 최종 답변 생성
            print("\n4. GPT-4o 최종 답변 생성...")
            llm_result = self.llm_processor.generate_bmw_manual_response(
                user_prompt=stt_text,
                manual_pages=similar_pages,
                segmented_part=segmented_path
            )
            
            total_time = time.time() - start_time
            
            # 결과 구성
            result = {
                'success': True,
                'stt_text': stt_text,
                'segmented_image_path': segmented_path,
                'processed_image_path': processed_path,
                'schrodinger_mode': schrodinger_mode,
                'top_manual_pages': similar_pages,
                'final_response': llm_result.get('response', 'Response generation failed'),
                'processing_time': total_time,
                'detailed_times': {
                    'sam2_time': sam2_result['inference_time'],
                    'total_time': total_time
                }
            }
            
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
        temp_files = glob.glob("temp_segmented_*.png")
        for file in temp_files:
            try:
                os.remove(file)
            except:
                pass

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
    test_cases = load_test_cases()
    
    if not test_cases:
        print("테스트 케이스가 없습니다. test_cases.json 파일을 확인하세요.")
        return
    
    try:
        # BMW 어시스턴트 초기화
        assistant = BMWManualAssistant()
        
        # 각 테스트 케이스 실행
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"테스트 케이스 {i}/{len(test_cases)}")
            print(f"{'='*60}")
            
            result = assistant.process_request(test_case)
            result['test_case_id'] = i
            results.append(result)
            
            if result['success']:
                print(f"\n✅ 테스트 {i} 성공")
            else:
                print(f"\n❌ 테스트 {i} 실패: {result['error']}")
        
        # 결과 저장
        with open('pipeline_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n테스트 결과 저장: pipeline_test_results.json")
        
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