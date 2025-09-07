import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# 루트 디렉토리 경로 설정
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)
unsb_dir = os.path.join(root_dir, 'UNSB')
if unsb_dir not in sys.path:
    sys.path.insert(0, unsb_dir)

# 필요한 프로세서들 import
from processors.colpali_processor import ColPaliProcessor
from processors.llm_processor import LLMProcessor

# 현재 디렉토리의 processors import  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from placeholder_processor import PlaceholderProcessor

class SimpleExperimentPipeline:
    def __init__(self, pdf_images_folder: str = "output/PDF_Images"):
        """단순화된 실험 파이프라인 초기화"""
        print("=== 단순화된 실험 파이프라인 초기화 ===")
        
        # 필요한 프로세서들만 초기화
        self.colpali_processor = ColPaliProcessor()
        self.llm_processor = LLMProcessor()
        self.placeholder_processor = PlaceholderProcessor()
        
        # ColPali 데이터베이스 구축
        print(f"ColPali 데이터베이스 초기화: {pdf_images_folder}")
        try:
            num_images, build_time = self.colpali_processor.build_database_from_pdf_images(
                pdf_images_folder, batch_size=8
            )
            print(f"ColPali 데이터베이스 구축 완료: {num_images}개 이미지, {build_time:.2f}초")
        except Exception as e:
            print(f"ColPali 데이터베이스 구축 실패: {e}")
            raise
        
        print("단순화된 실험 파이프라인 초기화 완료")
    
    def process_single_experiment(self, question: Dict[str, Any], 
                                output_dir: str) -> Dict[str, Any]:
        """
        단일 실험 실행 - JSON의 image_path에 따라 자동으로 모드 결정
        
        Args:
            question: 질문 데이터
            output_dir: 출력 디렉토리
            
        Returns:
            실험 결과 딕셔너리
        """
        start_time = time.time()
        
        # image_path 존재 여부로 모드 자동 결정
        image_path = question.get('image_path', '')
        if image_path and image_path.strip() and os.path.exists(image_path):
            input_type = "text_image"
        else:
            input_type = "text"
            image_path = None
        
        experiment_id = f"q{question['id']}_{input_type}"
        
        print(f"\n=== 실험 {experiment_id} 시작 ===")
        print(f"입력 타입: {input_type}")
        print(f"질문: {question['query_text']}")
        if image_path:
            print(f"이미지: {image_path}")
        
        try:
            # 1. 텍스트 처리 (placeholder 변환)
            processed_text = self._process_text(question['query_text'], input_type)
            print(f"처리된 텍스트: '{processed_text}'")
            
            # 2. ColPali 검색 - input_type에 따라 다른 방식
            if input_type == "text":
                # 텍스트만으로 검색
                print(f"텍스트 검색 실행")
                search_results = self.colpali_processor.search_by_text(processed_text, k=5)
            else:
                # 이미지로 검색
                print(f"이미지 검색 실행: {image_path}")
                search_results = self.colpali_processor.search_by_image(image_path, k=5)
            
            print(f"검색된 페이지 수: {len(search_results)}")
            
            # 3. LLM 답변 생성
            llm_response = self._generate_llm_response(
                processed_text, 
                search_results, 
                image_path  # text_image 모드일 때만 이미지 전달
            )
            
            total_time = time.time() - start_time
            
            # 결과 구성
            result = {
                'experiment_id': experiment_id,
                'question_id': question['id'],
                'input_type': input_type,
                'original_text': question['query_text'],
                'processed_text': processed_text,
                'query_image_path': image_path,
                'search_results': search_results,
                'llm_response': llm_response.get('response', ''),
                'expected_answer': question.get('expected_answer', ''),
                'source_page': question.get('source_page', ''),
                'processing_time': total_time,
                'success': True
            }
            
            print(f"실험 {experiment_id} 완료 ({total_time:.2f}초)")
            return result
            
        except Exception as e:
            error_result = {
                'experiment_id': experiment_id,
                'question_id': question['id'],
                'input_type': input_type,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
            print(f"실험 {experiment_id} 실패: {e}")
            return error_result
    
    def _process_text(self, text: str, input_type: str) -> str:
        """
        텍스트 처리
        
        Args:
            text: 원본 텍스트
            input_type: "text" 또는 "text_image"
            
        Returns:
            처리된 텍스트
        """
        return self.placeholder_processor.process_text(text, input_type)
    
    def _generate_llm_response(self, text: str, search_results: List[Dict[str, Any]], 
                              image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        LLM 답변 생성
        
        Args:
            text: 처리된 텍스트
            search_results: ColPali 검색 결과
            image_path: 크롭된 이미지 경로 (text_image 모드에서만)
            
        Returns:
            LLM 응답 딕셔너리
        """
        try:
            return self.llm_processor.generate_bmw_manual_response(
                user_prompt=text,
                manual_pages=search_results,
                segmented_part=image_path  # 크롭된 이미지
            )
        except Exception as e:
            print(f"LLM 답변 생성 실패: {e}")
            return {'response': f'답변 생성 실패: {e}'}
    
    def run_experiments(self, questions_file: str,  
                       output_dir: str = "Paper_Exp/output/simple_experiments") -> Dict[str, Any]:
        """
        전체 실험 실행
        
        Args:
            questions_file: 질문 JSON 파일 경로
            output_dir: 출력 디렉토리
            
        Returns:
            실험 결과 정보
        """
        # 질문 데이터 로드
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = data['questions']
        
        # JSON 파일명을 기반으로 실험 폴더명 생성
        json_filename = Path(questions_file).stem  # 확장자 제거한 파일명
        exp_dir = os.path.join(output_dir, json_filename)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"\n=== 단순화된 실험 시작 ===")
        print(f"질문 파일: {questions_file}")
        print(f"질문 수: {len(questions)}")
        print(f"출력 디렉토리: {exp_dir}")
        
        # 실험 실행
        all_results = []
        
        for question in questions:
            result = self.process_single_experiment(question, exp_dir)
            all_results.append(result)
        
        # 결과 저장
        results_file = os.path.join(exp_dir, "all_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n=== 단순화된 실험 완료 ===")
        print(f"실험 폴더: {exp_dir}")
        print(f"결과 저장: {results_file}")
        print(f"총 {len(all_results)}개 실험 완료")
        
        return {"results_file": results_file, "total_experiments": len(all_results)}

def main():
    """테스트용 메인 함수"""
    try:
        # 파이프라인 초기화
        pipeline = SimpleExperimentPipeline()
        
        # 테스트 실행
        return pipeline.run_experiments(
            questions_file="Paper_Exp/Questions/Simple_Questions.json"
        )
        
    except Exception as e:
        print(f"테스트 실행 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()