import os
import sys
import argparse
from pathlib import Path
import json

# 프로젝트 루트 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from Paper_Exp.scripts.simple_experiment_pipeline import SimpleExperimentPipeline
from Paper_Exp.scripts.ragchecker_converter import RAGCheckerConverter

def validate_environment():
    """실험 환경 검증"""
    print("=== 환경 검증 ===")
    
    required_paths = [
        "Paper_Exp/Questions",
        "Paper_Exp/Query_images", 
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

def run_simple_experiments(args):
    """단순화된 실험 실행"""
    print("\n=== 단순화된 실험 실행 ===")
    
    # 실험 설정 출력
    print(f"질문 파일: {args.questions}")
    print(f"출력 디렉토리: {args.output}")
    
    try:
        # 실험 실행
        pipeline = SimpleExperimentPipeline(pdf_images_folder=args.pdf_images)
        
        result = pipeline.run_experiments(
            questions_file=args.questions,
            output_dir=args.output
        )
        
        print(f"\n실험 완료! 총 {result['total_experiments']}개 실험")
        
        # RAGChecker 변환
        if args.convert_ragchecker:
            print("\n=== RAGChecker 변환 ===")
            converter = RAGCheckerConverter()
            
            # 결과 디렉토리에서 RAGChecker 파일 생성 (원본과 별도 저장)
            results_dir = os.path.dirname(result['results_file'])
            ragchecker_file = os.path.join(results_dir, "ragchecker_results.json")
            
            # RAGChecker 형태로 변환 (원본은 보존)
            converter.convert_experiment_results(
                result['results_file'],  # 원본: all_results.json
                ragchecker_file          # 변환: ragchecker_results.json
            )
            
            print(f"원본 결과: {result['results_file']}")
            print(f"RAGChecker 변환: {ragchecker_file}")
            print("두 파일 모두 보존됨")
        
        return True
        
    except Exception as e:
        print(f"실험 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    os.chdir(PROJECT_ROOT)
    print(f"현재 작업 디렉토리 변경: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(
        description="단순화된 BMW Manual Assistant 실험 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행
  python simple_run_experiments.py
  
  # RAGChecker 변환 포함
  python simple_run_experiments.py --convert-ragchecker
  
  # 사용자 정의 설정
  python simple_run_experiments.py --questions Simple_Questions.json --output results/
        """
    )
    
    # 기본값 설정
    default_questions = "Paper_Exp/Questions/Simple_Questions.json"
    default_output = "Paper_Exp/output/simple_experiments"
    default_pdf_images = "output/PDF_Images"
    
    parser.add_argument("--questions", default=default_questions,
                       help=f"질문 JSON 파일 경로 (기본값: {default_questions})")
    
    parser.add_argument("--output", default=default_output,
                       help=f"출력 디렉토리 (기본값: {default_output})")
    
    parser.add_argument("--pdf_images", default=default_pdf_images,
                       help=f"PDF 이미지 디렉토리 (기본값: {default_pdf_images})")
    
    # 실행 옵션
    parser.add_argument("--convert-ragchecker", action="store_true",
                       help="실험 완료 후 RAGChecker 형태로 변환")
    
    parser.add_argument("--validate-only", action="store_true",
                       help="환경 검증만 실행")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("단순화된 BMW Manual Assistant 실험 파이프라인")
    print("Text only & Text+Image 모드")
    print("=" * 60)
    
    # 환경 검증
    if not validate_environment():
        print("\n환경 검증 실패. 필요한 폴더와 파일을 확인하세요.")
        return 1
    
    if args.validate_only:
        print("\n환경 검증만 완료.")
        return 0
    
    # 실험 실행
    if run_simple_experiments(args):
        print("\n모든 실험 완료!")
        return 0
    else:
        print("\n실험 실행 실패!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)