import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
from Paper_Exp.scripts.experiment_pipeline import ExperimentPipeline
from Paper_Exp.scripts.ragchecker_converter import RAGCheckerConverter

def validate_environment():
    """실험 환경 검증"""
    print("=== 환경 검증 ===")
    
    required_paths = [
        "Paper_Exp/Questions",
        "Paper_Exp/Query_images", 
        "Paper_Exp/output",
        "Paper_Exp/scripts",
        "output/PDF_Images",
        "processors"     
    ]
        
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ 필수 경로가 없습니다: {missing_paths}")
        return False
    
    print("✅ 환경 검증 완료")
    return True

def run_quick_test():
    """빠른 테스트 실행 (1개 질문, 1개 파이프라인)"""
    print("\n=== 빠른 테스트 실행 ===")
    
    try:
        pipeline = ExperimentPipeline()
        
        # 첫 번째 질문만, none 파이프라인만, text 타입만
        summary = pipeline.run_experiments(
            questions_file="Questions/questions.json",
            input_types=["text"],
            pipeline_ids=["none"],
            output_dir="output/quick_test"
        )
        
        print(f"빠른 테스트 완료! 성공률: {summary['success_rate']:.1f}%")
        return True
        
    except Exception as e:
        print(f"빠른 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_experiments(args):
    """전체 실험 실행"""
    print("\n=== 전체 실험 실행 ===")
    
    # 실험 설정 출력
    print(f"질문 파일: {args.questions}")
    print(f"입력 타입: {args.input_types}")
    print(f"파이프라인: {args.pipelines if args.pipelines else '전체'}")
    print(f"출력 디렉토리: {args.output}")
    
    try:
        # 실험 실행
        pipeline = ExperimentPipeline(pdf_images_folder=args.pdf_images)
        
        summary = pipeline.run_experiments(
            questions_file=args.questions,
            input_types=args.input_types,
            pipeline_ids=args.pipelines,
            output_dir=args.output
        )
        
        print(f"\n전체 실험 완료! 성공률: {summary['success_rate']:.1f}%")
        
        # 결과 디렉토리 찾기
        experiment_dirs = [d for d in os.listdir(args.output) if d.startswith("experiment_")]
        if experiment_dirs:
            latest_exp_dir = os.path.join(args.output, sorted(experiment_dirs)[-1])
            
            # RAGChecker 변환
            if args.convert_ragchecker:
                print("\n=== RAGChecker 변환 ===")
                converter = RAGCheckerConverter()
                
                ragchecker_output_dir = os.path.join(latest_exp_dir, "ragchecker")
                output_files = converter.convert_by_pipeline(latest_exp_dir, ragchecker_output_dir)
                
                print("RAGChecker 파일 생성 완료:")
                for pipeline, file_path in output_files.items():
                    print(f"  {pipeline}: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"실험 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    os.chdir(PROJECT_ROOT)
    print(f"현재 작업 디렉토리 변경: {os.getcwd()}") # 확인용 출력
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="ILLIT (Image-and-Language Lookup Interactive Toolkit) 실험 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 빠른 테스트
  python run_experiments.py --quick-test
  
  # 특정 파이프라인만 실행
  python run_experiments.py --pipelines none bilateral --input_types text text_image
  
  # 전체 실험 + RAGChecker 변환
  python run_experiments.py --convert-ragchecker
  
  # 사용자 정의 설정
  python run_experiments.py --questions custom.json --output results/ --pdf_images ../custom_pdfs/
        """
    )
    
    default_questions = "Paper_Exp/Questions/questions.json"
    default_output = "Paper_Exp/output/experiments"
    default_pdf_images = "output/PDF_Images" # 이 경로는 루트 기준이므로 그대로.
    
    parser.add_argument("--questions", default=default_questions,
                       help=f"질문 JSON 파일 경로 (기본값: {default_questions})")
    
    parser.add_argument("--input_types", nargs="+", 
                       default=["text", "image", "text_image"],
                       choices=["text", "image", "text_image"],
                       help="실행할 입력 타입 (기본값: 전체)")
    
    parser.add_argument("--pipelines", nargs="+", 
                       choices=["none", "schrodinger", "sam2", "bilateral", 
                               "schrodinger_sam2", "bilateral_sam2"],
                       help="실행할 파이프라인 (기본값: 전체)")
    
    parser.add_argument("--output", default=default_output,
                       help=f"출력 디렉토리 (기본값: {default_output})")
    
    parser.add_argument("--pdf_images", default=default_pdf_images,
                       help=f"PDF 이미지 디렉토리 (기본값: {default_pdf_images})")
    
    # 실행 옵션
    parser.add_argument("--quick-test", action="store_true",
                       help="빠른 테스트 실행 (1개 질문, none 파이프라인, text 타입)")
    
    parser.add_argument("--convert-ragchecker", action="store_true",
                       help="실험 완료 후 RAGChecker 형태로 변환")
    
    parser.add_argument("--validate-only", action="store_true",
                       help="환경 검증만 실행")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ILLIT (Image-and-Language Lookup Interactive Toolkit)")
    print("BMW Manual Assistant 실험 파이프라인")
    print("=" * 60)
    
    # 환경 검증
    if not validate_environment():
        print("\n환경 검증 실패. 필요한 폴더와 파일을 확인하세요.")
        return 1
    
    if args.validate_only:
        print("\n환경 검증만 완료.")
        return 0
    
    # 빠른 테스트
    if args.quick_test:
        if run_quick_test():
            print("\n빠른 테스트 성공!")
            return 0
        else:
            print("\n빠른 테스트 실패!")
            return 1
    
    # 전체 실험
    if run_full_experiments(args):
        print("\n모든 실험 완료!")
        return 0
    else:
        print("\n실험 실행 실패!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)