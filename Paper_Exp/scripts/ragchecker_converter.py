import json
import os
import sys
from typing import Dict, List, Any
from pathlib import Path

# 상위 디렉토리의 processors import
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from processors.llm_processor import LLMProcessor

class RAGCheckerConverter:
    def __init__(self):
        """RAGChecker 변환기 초기화"""
        self.llm_processor = LLMProcessor()
        print("RAGChecker 변환기 초기화 완료")
    
    def convert_experiment_results(self, experiment_results_file: str, output_file: str) -> str:
        """
        실험 결과를 RAGChecker 형태로 변환
        
        Args:
            experiment_results_file: 실험 결과 JSON 파일
            output_file: RAGChecker 형태 출력 파일
            
        Returns:
            str: 변환된 파일 경로
        """
        print(f"실험 결과 변환 시작: {experiment_results_file}")
        
        # 실험 결과 로드
        with open(experiment_results_file, 'r', encoding='utf-8') as f:
            experiment_results = json.load(f)
        
        # RAGChecker 형태로 변환
        ragchecker_results = []
        
        for result in experiment_results:
            if not result.get('success', False):
                print(f"실패한 실험 건너뛰기: {result.get('experiment_id', 'unknown')}")
                continue
            
            ragchecker_item = self._convert_single_result(result)
            if ragchecker_item:
                ragchecker_results.append(ragchecker_item)
        
        # RAGChecker 형태로 저장
        ragchecker_data = {
            "results": ragchecker_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ragchecker_data, f, indent=2, ensure_ascii=False)
        
        print(f"RAGChecker 변환 완료: {len(ragchecker_results)}개 결과 → {output_file}")
        return output_file
    
    def _convert_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 실험 결과를 RAGChecker 형태로 변환
        
        Args:
            result: 실험 결과 딕셔너리
            
        Returns:
            Dict: RAGChecker 형태 딕셔너리
        """
        try:
            # query_id 생성 (실험 ID 사용)
            query_id = result.get('experiment_id', f"exp_{result.get('question_id', 'unknown')}")
            
            # query 텍스트 (처리된 텍스트 사용)
            query = result.get('processed_text', result.get('original_text', ''))
            
            # gt_answer (기대 답변)
            gt_answer = result.get('expected_answer', '')
            
            # response (LLM 생성 답변)
            response = result.get('llm_response', '')
            
            # retrieved_context 생성
            retrieved_context = self._convert_search_results_to_context(
                result.get('search_results', [])
            )
            
            ragchecker_item = {
                "query_id": query_id,
                "query": query,
                "gt_answer": gt_answer,
                "response": response,
                "retrieved_context": retrieved_context
            }
            
            return ragchecker_item
            
        except Exception as e:
            print(f"결과 변환 실패: {e}")
            return None
    
    def _convert_search_results_to_context(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        ColPali 검색 결과를 RAGChecker retrieved_context로 변환
        
        Args:
            search_results: ColPali 검색 결과 리스트
            
        Returns:
            List[Dict]: RAGChecker retrieved_context 형태
        """
        retrieved_context = []
        
        for i, result in enumerate(search_results):
            try:
                # doc_id는 이미지 파일명 기반
                doc_id = result.get('image_name', f"manual_page_{i}")
                if doc_id.endswith('.png') or doc_id.endswith('.jpg'):
                    doc_id = os.path.splitext(doc_id)[0]  # 확장자 제거
                
                # text는 이미지에서 텍스트 추출
                text_content = self._extract_text_from_manual_image(
                    result.get('image_path', '')
                )
                
                context_item = {
                    "doc_id": doc_id,
                    "text": text_content
                }
                
                retrieved_context.append(context_item)
                
            except Exception as e:
                print(f"검색 결과 변환 실패: {e}")
                continue
        
        return retrieved_context
    
    def _extract_text_from_manual_image(self, image_path: str) -> str:
        """
        매뉴얼 이미지에서 텍스트 추출
        
        Args:
            image_path: 이미지 경로
            
        Returns:
            str: 추출된 텍스트
        """
        if not image_path or not os.path.exists(image_path):
            return f"이미지 경로 없음: {image_path}"
        
        try:
            # LLM을 사용해서 이미지에서 텍스트 추출
            extraction_prompt = """
            이 BMW 매뉴얼 이미지에서 모든 텍스트를 추출해주세요.
            다음 형식으로 정리해주세요:
            - 제목이나 헤더가 있으면 먼저 작성
            - 본문 내용을 순서대로 작성
            - 버튼명, 메뉴명 등도 포함
            - 한글과 영어 모두 포함
            
            텍스트만 추출하고 설명은 추가하지 마세요.
            """
            
            result = self.llm_processor.generate_bmw_manual_response(
                user_prompt=extraction_prompt,
                manual_pages=[],  # 빈 리스트
                segmented_part=image_path
            )
            
            extracted_text = result.get('response', '')
            
            # 텍스트가 너무 길면 요약
            if len(extracted_text) > 500:
                extracted_text = extracted_text[:500] + "..."
            
            return extracted_text
            
        except Exception as e:
            print(f"텍스트 추출 실패 {image_path}: {e}")
            return f"텍스트 추출 실패: {str(e)}"
    
    def convert_by_pipeline(self, experiment_dir: str, output_dir: str) -> Dict[str, str]:
        """
        파이프라인별로 RAGChecker 파일 생성
        
        Args:
            experiment_dir: 실험 결과 디렉토리
            output_dir: RAGChecker 출력 디렉토리
            
        Returns:
            Dict[str, str]: 파이프라인명 → 파일 경로 매핑
        """
        # 실험 결과 로드
        results_file = os.path.join(experiment_dir, "all_results.json")
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"실험 결과 파일을 찾을 수 없음: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        # 파이프라인별로 그룹핑
        pipeline_groups = {}
        for result in all_results:
            if result.get('success', False):
                pipeline = result.get('pipeline', 'unknown')
                if pipeline not in pipeline_groups:
                    pipeline_groups[pipeline] = []
                pipeline_groups[pipeline].append(result)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파이프라인별 RAGChecker 파일 생성
        output_files = {}
        for pipeline, results in pipeline_groups.items():
            # 파일명 안전하게 변환
            safe_pipeline_name = pipeline.replace("+", "_").replace(" ", "_").lower()
            output_file = os.path.join(output_dir, f"ragchecker_{safe_pipeline_name}.json")
            
            # RAGChecker 형태로 변환
            ragchecker_results = []
            for result in results:
                ragchecker_item = self._convert_single_result(result)
                if ragchecker_item:
                    ragchecker_results.append(ragchecker_item)
            
            # 저장
            ragchecker_data = {"results": ragchecker_results}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ragchecker_data, f, indent=2, ensure_ascii=False)
            
            output_files[pipeline] = output_file
            print(f"파이프라인 '{pipeline}': {len(ragchecker_results)}개 결과 → {output_file}")
        
        return output_files

def test_ragchecker_converter():
    """RAGChecker 변환기 테스트"""
    converter = RAGCheckerConverter()
    
    # 테스트용 실험 결과 생성
    test_results = [
        {
            "experiment_id": "q1_none_text",
            "question_id": 1,
            "pipeline": "원본",
            "input_type": "text",
            "original_text": "{분할 스크린}에서 설정하고...",
            "processed_text": "분할 스크린에서 설정하고...",
            "llm_response": "분할 스크린에서는 지도, 교통상황 등을 설정할 수 있습니다.",
            "expected_answer": "분할 스크린에서는 지도, 교통상황, 교통사건 등을 표시할 수 있습니다.",
            "search_results": [
                {
                    "image_name": "manual_page_25.png",
                    "image_path": "../Query_images/3.png",
                    "similarity_score": 0.85,
                    "rank": 1
                }
            ],
            "success": True
        }
    ]
    
    # 테스트 파일 저장
    test_file = "test_results.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    try:
        # 변환 테스트
        output_file = converter.convert_experiment_results(test_file, "test_ragchecker.json")
        print(f"테스트 변환 완료: {output_file}")
        
        # 결과 확인
        with open(output_file, 'r', encoding='utf-8') as f:
            ragchecker_data = json.load(f)
        
        print("RAGChecker 형태 확인:")
        print(json.dumps(ragchecker_data, indent=2, ensure_ascii=False))
        
        # 테스트 파일 정리
        os.remove(test_file)
        os.remove(output_file)
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        # 정리
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_ragchecker_converter()