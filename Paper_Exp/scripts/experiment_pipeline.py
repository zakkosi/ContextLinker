import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)
unsb_dir = os.path.join(root_dir, 'UNSB')
if unsb_dir not in sys.path:
    sys.path.insert(0, unsb_dir)

# main.py와 동일하게 import
from processors.sam2_processor import SAM2Segmenter
from processors.colpali_processor import ColPaliProcessor
from processors.llm_processor import LLMProcessor

# 현재 디렉토리의 processors import  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bilateral_processor import BilateralFilterProcessor
from placeholder_processor import PlaceholderProcessor

class ExperimentPipeline:
    def __init__(self, pdf_images_folder: str = "../../output/PDF_Images"):
        """실험 파이프라인 초기화"""
        print("=== 실험 파이프라인 초기화 ===")
        
        # 프로세서들 초기화
        self.sam2_processor = SAM2Segmenter()
        self.colpali_processor = ColPaliProcessor()
        self.llm_processor = LLMProcessor()
        
        # *** 핵심 수정: SchrodingerBridge 초기화 시 sys.argv 격리 ***
        self.schrodinger_processor = self._initialize_schrodinger_safely()
        
        self.bilateral_processor = BilateralFilterProcessor()
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
        
        print("실험 파이프라인 초기화 완료")
    
    def _initialize_schrodinger_safely(self):
        """
        SchrodingerBridge를 sys.argv 충돌 없이 안전하게 초기화
        """
        print("SchrodingerBridge 프로세서 안전 초기화 중...")
        
        # 현재 sys.argv 백업
        original_argv = sys.argv.copy()
        
        try:
            # SchrodingerBridge용 임시 argv 설정 (UNSB가 기대하는 형태)
            sys.argv = [
                'experiment_pipeline.py',  # 스크립트 이름
                '--name', '225_net_G_B2A',
                '--checkpoints_dir', 'models/checkpoints', 
                '--epoch', 'latest',
                '--model', 'sb',
                '--phase', 'test',
                '--batch_size', '1',
                '--eval',
                '--gpu_ids', '0'
            ]
            
            # 이제 안전하게 SchrodingerBridge import 및 초기화
            from processors.schrodinger_processor import SchrodingerBridgeProcessor
            schrodinger_processor = SchrodingerBridgeProcessor()
            
            print("SchrodingerBridge 프로세서 안전 초기화 완료")
            return schrodinger_processor
            
        except Exception as e:
            print(f"SchrodingerBridge 초기화 실패: {e}")
            print("SchrodingerBridge 없이 계속 진행합니다.")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # 반드시 원래 argv 복원
            sys.argv = original_argv
            print(f"sys.argv 복원 완료: {sys.argv}")
    
    def get_pipeline_combinations(self) -> List[Dict[str, Any]]:
        """6가지 파이프라인 조합 정의"""
        return [
            {"id": "none", "name": "원본", "use_schrodinger": False, "use_sam2": False, "use_bilateral": False},
            {"id": "schrodinger", "name": "Schrodingerㅇ만", "use_schrodinger": True, "use_sam2": False, "use_bilateral": False},
            {"id": "sam2", "name": "SAM2만", "use_schrodinger": False, "use_sam2": True, "use_bilateral": False},
            {"id": "bilateral", "name": "Bilateral만", "use_schrodinger": False, "use_sam2": False, "use_bilateral": True},
            {"id": "schrodinger_sam2", "name": "Schrodinger+SAM2", "use_schrodinger": True, "use_sam2": True, "use_bilateral": False},
            {"id": "bilateral_sam2", "name": "Bilateral+SAM2", "use_schrodinger": False, "use_sam2": True, "use_bilateral": True}
        ]
    
    def process_single_experiment(self, question: Dict[str, Any], pipeline_config: Dict[str, Any], 
                                input_type: str, output_dir: str) -> Dict[str, Any]:
        """단일 실험 실행"""
        start_time = time.time()
        experiment_id = f"q{question['id']}_{pipeline_config['id']}_{input_type}"
        
        print(f"\n=== 실험 {experiment_id} 시작 ===")
        print(f"파이프라인: {pipeline_config['name']}")
        print(f"입력 타입: {input_type}")
        
        try:
            # 1. 텍스트 처리 (placeholder 변환)
            processed_text = self._process_text(question['query_text'], input_type)
            print(f"처리된 텍스트: '{processed_text}'")
            
            # 2. 이미지 처리
            processed_image_path = self._process_image(
                question.get('image_path', ''), 
                pipeline_config, 
                input_type,
                output_dir,
                experiment_id
            )
            
            # 3. ColPali 검색
            search_results = self._search_manual_pages(processed_text, processed_image_path, input_type)
            
            # 4. LLM 답변 생성
            llm_response = self._generate_llm_response(processed_text, search_results, processed_image_path)
            
            total_time = time.time() - start_time
            
            # 결과 구성
            result = {
                'experiment_id': experiment_id,
                'question_id': question['id'],
                'pipeline': pipeline_config['name'],
                'input_type': input_type,
                'original_text': question['query_text'],
                'processed_text': processed_text,
                'processed_image_path': processed_image_path,
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
                'pipeline': pipeline_config['name'],
                'input_type': input_type,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
            print(f"실험 {experiment_id} 실패: {e}")
            return error_result
    
    def _process_text(self, text: str, input_type: str) -> str:
        """텍스트 처리"""
        return self.placeholder_processor.process_text(text, input_type)
    
    def _process_image(self, image_path: str, pipeline_config: Dict[str, Any], 
                      input_type: str, output_dir: str, experiment_id: str) -> Optional[str]:
        """이미지 처리"""
        if input_type == "text":
            return None
        
        if not image_path or not os.path.exists(image_path):
            print(f"이미지를 찾을 수 없음: {image_path}")
            return None
        
        current_image = image_path
        
        # Schrodinger 처리
        if pipeline_config.get('use_schrodinger', False):
            if self.schrodinger_processor is not None:
                try:
                    schrodinger_path = self.schrodinger_processor.real_to_illustration(
                        current_image, 
                        os.path.join(output_dir, experiment_id)
                    )
                    current_image = schrodinger_path
                    print(f"Schrodinger 처리 완료: {schrodinger_path}")
                except Exception as e:
                    print(f"Schrodinger 처리 실패: {e}, 원본 이미지 사용")
            else:
                print("SchrodingerBridge 프로세서가 없습니다. 스킵합니다.")
                # Schrodinger가 없으면 이 파이프라인 자체를 실패로 처리
                raise Exception("SchrodingerBridge 프로세서가 필요하지만 초기화되지 않았습니다.")
        
        # Bilateral 처리  
        if pipeline_config.get('use_bilateral', False):
            bilateral_path = self.bilateral_processor.manual_style_filter(
                current_image,
                os.path.join(output_dir, experiment_id)
            )
            current_image = bilateral_path
            print(f"Bilateral 처리 완료: {bilateral_path}")
        
        # SAM2 처리
        if pipeline_config.get('use_sam2', False):
            sam2_result = self.sam2_processor.process_image(current_image, mode=0)
            sam2_path = os.path.join(output_dir, experiment_id, f"sam2_{experiment_id}.png")
            os.makedirs(os.path.dirname(sam2_path), exist_ok=True)
            self.sam2_processor.save_result(sam2_result, sam2_path)
            current_image = sam2_path
            print(f"SAM2 처리 완료: {sam2_path}")
        
        return current_image
    
    def _search_manual_pages(self, text: str, image_path: Optional[str], input_type: str) -> List[Dict[str, Any]]:
        """ColPali로 매뉴얼 페이지 검색"""
        try:
            if input_type == "text" and text:
                # 텍스트 검색
                return self.colpali_processor.search_by_text(text, k=5)
            elif input_type == "image" and image_path:
                # 이미지 검색
                return self.colpali_processor.search_by_image(image_path, k=5)
            elif input_type == "text_image" and image_path:
                # 이미지 검색 우선 (텍스트는 LLM에서 참고)
                return self.colpali_processor.search_by_image(image_path, k=5)
            else:
                return []
        except Exception as e:
            print(f"검색 실패: {e}")
            return []
    
    def _generate_llm_response(self, text: str, search_results: List[Dict[str, Any]], 
                              image_path: Optional[str]) -> Dict[str, Any]:
        """LLM 답변 생성"""
        try:
            return self.llm_processor.generate_bmw_manual_response(
                user_prompt=text,
                manual_pages=search_results,
                segmented_part=image_path
            )
        except Exception as e:
            print(f"LLM 답변 생성 실패: {e}")
            return {'response': f'답변 생성 실패: {e}'}
    
    def run_experiments(self, questions_file: str, input_types: List[str], 
                       pipeline_ids: Optional[List[str]] = None, 
                       output_dir: str = "../output/experiments") -> Dict[str, Any]:
        """전체 실험 실행"""
        # 질문 데이터 로드
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = data['questions']
        
        # 파이프라인 조합 가져오기
        all_pipelines = self.get_pipeline_combinations()
        if pipeline_ids:
            pipelines = [p for p in all_pipelines if p['id'] in pipeline_ids]
        else:
            pipelines = all_pipelines
        
        # 실험 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(output_dir, f"experiment_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"\n=== 실험 시작 ===")
        print(f"질문 수: {len(questions)}")
        print(f"파이프라인: {[p['name'] for p in pipelines]}")
        print(f"입력 타입: {input_types}")
        print(f"총 실험 수: {len(questions) * len(pipelines) * len(input_types)}")
        print(f"출력 디렉토리: {exp_dir}")
        
        # 실험 실행
        all_results = []
        total_experiments = len(questions) * len(pipelines) * len(input_types)
        current_exp = 0
        
        for question in questions:
            for pipeline in pipelines:
                for input_type in input_types:
                    current_exp += 1
                    print(f"\n진행률: {current_exp}/{total_experiments}")
                    
                    result = self.process_single_experiment(
                        question, pipeline, input_type, exp_dir
                    )
                    all_results.append(result)
        
        # 결과 저장
        results_file = os.path.join(exp_dir, "all_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 요약 생성
        summary = self._generate_summary(all_results)
        summary_file = os.path.join(exp_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 전체 실험 완료 후 최종 정리
        self._final_cleanup()
        
        print(f"\n=== 실험 완료 ===")
        print(f"결과 저장: {results_file}")
        print(f"요약 저장: {summary_file}")
        print(f"성공률: {summary['success_rate']:.1f}%")
        
        return summary
    
    def _final_cleanup(self):
        """실험 완료 후 최종 정리"""
        try:
            # temp 디렉토리 완전 삭제
            import shutil
            if os.path.exists("temp"):
                shutil.rmtree("temp")
                print("temp 디렉토리 완전 삭제")
            
            # 기타 임시 파일 패턴 정리
            import glob
            temp_patterns = [
                "temp_*.*",
                "bilateral_filtered_*.*",
                "real_to_illust_*.*"
            ]
            
            for pattern in temp_patterns:
                for temp_file in glob.glob(pattern):
                    try:
                        os.remove(temp_file)
                        print(f"임시 파일 정리: {temp_file}")
                    except:
                        pass
                        
        except Exception as e:
            print(f"최종 정리 중 오류: {e}")
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """실험 결과 요약 생성"""
        total = len(results)
        successful = len([r for r in results if r.get('success', False)])
        
        # 파이프라인별 성공률
        pipeline_stats = {}
        input_type_stats = {}
        
        for result in results:
            pipeline = result.get('pipeline', 'unknown')
            input_type = result.get('input_type', 'unknown')
            success = result.get('success', False)
            
            if pipeline not in pipeline_stats:
                pipeline_stats[pipeline] = {'total': 0, 'success': 0}
            pipeline_stats[pipeline]['total'] += 1
            if success:
                pipeline_stats[pipeline]['success'] += 1
            
            if input_type not in input_type_stats:
                input_type_stats[input_type] = {'total': 0, 'success': 0}
            input_type_stats[input_type]['total'] += 1
            if success:
                input_type_stats[input_type]['success'] += 1
        
        # 성공률 계산
        for stats in pipeline_stats.values():
            stats['success_rate'] = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        
        for stats in input_type_stats.values():
            stats['success_rate'] = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        
        return {
            'total_experiments': total,
            'successful_experiments': successful,
            'success_rate': (successful / total) * 100 if total > 0 else 0,
            'pipeline_stats': pipeline_stats,
            'input_type_stats': input_type_stats,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="BMW Manual Assistant 실험 파이프라인")
    
    parser.add_argument("--questions", default="../Questions/questions.json",
                       help="질문 JSON 파일 경로")
    parser.add_argument("--input_types", nargs="+", default=["text", "image", "text_image"],
                       choices=["text", "image", "text_image"],
                       help="실행할 입력 타입")
    parser.add_argument("--pipelines", nargs="+", 
                       choices=["none", "schrodinger", "sam2", "bilateral", "schrodinger_sam2", "bilateral_sam2"],
                       help="실행할 파이프라인 (전체 실행시 생략)")
    parser.add_argument("--output", default="../output/experiments",
                       help="출력 디렉토리")
    parser.add_argument("--pdf_images", default="../../output/PDF_Images",
                       help="PDF 이미지 디렉토리")
    
    args = parser.parse_args()
    
    try:
        # 파이프라인 초기화
        pipeline = ExperimentPipeline(pdf_images_folder=args.pdf_images)
        
        # 실험 실행
        summary = pipeline.run_experiments(
            questions_file=args.questions,
            input_types=args.input_types,
            pipeline_ids=args.pipelines,
            output_dir=args.output
        )
        
        print(f"\n전체 실험 완료! 성공률: {summary['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"실험 실행 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()