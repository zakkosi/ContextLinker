import os
from PIL import Image
from pathlib import Path
import time
from typing import List, Dict, Any

def flip_image_horizontal(image_path: str, 
                         output_dir: str = None,
                         suffix: str = "_flipped") -> str:
    """
    이미지를 좌우 대칭으로 플립
    
    Args:
        image_path: 원본 이미지 경로
        output_dir: 출력 디렉토리 (None이면 원본과 같은 폴더)
        suffix: 파일명 뒤에 붙을 접미사
        
    Returns:
        str: 플립된 이미지 저장 경로
    """
    # 이미지 로드
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"이미지 로딩 실패 {image_path}: {e}")
    
    # 좌우 플립
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 출력 경로 설정
    path = Path(image_path)
    if output_dir is None:
        output_dir = path.parent
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # 출력 파일명 생성
    output_filename = f"{path.stem}{suffix}{path.suffix}"
    output_path = os.path.join(output_dir, output_filename)
    
    # 저장
    flipped_image.save(output_path)
    
    print(f"플립 완료: {path.name} → {output_filename}")
    return output_path

def flip_folder_images(input_dir: str,
                      output_dir: str = None,
                      suffix: str = "_flipped") -> List[Dict[str, Any]]:
    """
    폴더 내 모든 이미지를 좌우 플립
    
    Args:
        input_dir: 입력 이미지 폴더
        output_dir: 출력 폴더 (None이면 각 이미지와 같은 폴더)
        suffix: 파일명 접미사
        
    Returns:
        list: 처리 결과 리스트
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"입력 폴더를 찾을 수 없습니다: {input_dir}")
    
    # 지원되는 이미지 확장자
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not image_files:
        print(f"지원되는 이미지 파일이 없습니다: {input_dir}")
        return []
    
    print(f"플립할 이미지: {len(image_files)}개")
    
    results = []
    start_time = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 처리 중: {image_file.name}")
        
        try:
            output_path = flip_image_horizontal(
                str(image_file), 
                output_dir, 
                suffix
            )
            
            results.append({
                'original_path': str(image_file),
                'flipped_path': output_path,
                'original_name': image_file.name,
                'flipped_name': Path(output_path).name,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"처리 실패 {image_file.name}: {e}")
            results.append({
                'original_path': str(image_file),
                'original_name': image_file.name,
                'status': 'failed',
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    successful = len([r for r in results if r['status'] == 'success'])
    
    print(f"\n=== 플립 처리 완료 ===")
    print(f"성공: {successful}/{len(image_files)}")
    print(f"처리 시간: {total_time:.2f}초")
    
    return results

def flip_pdf_images(pdf_images_dir: str = "../output/PDF_Images") -> Dict[str, Any]:
    """
    PDF에서 추출된 이미지들을 모두 좌우 플립
    (데이터 증강 목적)
    
    Args:
        pdf_images_dir: PDF 이미지들이 있는 기본 디렉토리
        
    Returns:
        dict: 전체 처리 결과
    """
    pdf_dir = Path(pdf_images_dir)
    if not pdf_dir.exists():
        raise ValueError(f"PDF 이미지 폴더를 찾을 수 없습니다: {pdf_images_dir}")
    
    # 하위 폴더들 찾기 (각 PDF별 폴더)
    pdf_folders = [f for f in pdf_dir.iterdir() if f.is_dir()]
    
    if not pdf_folders:
        print(f"PDF 폴더가 없습니다: {pdf_images_dir}")
        return {'total_processed': 0, 'results': []}
    
    print(f"처리할 PDF 폴더: {len(pdf_folders)}개")
    
    all_results = []
    total_processed = 0
    
    for pdf_folder in pdf_folders:
        print(f"\n처리 중: {pdf_folder.name}")
        
        # 각 PDF 폴더 내 이미지들 플립
        folder_results = flip_folder_images(
            str(pdf_folder),
            output_dir=None,  # 같은 폴더에 저장
            suffix="_flipped"
        )
        
        successful_count = len([r for r in folder_results if r['status'] == 'success'])
        total_processed += successful_count
        
        all_results.extend(folder_results)
        
        print(f"{pdf_folder.name}: {successful_count}개 완료")
    
    return {
        'total_pdf_folders': len(pdf_folders),
        'total_processed': total_processed,
        'results': all_results
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("1. 단일 이미지: python image_flipper.py single <이미지경로>")
        print("2. 폴더 처리: python image_flipper.py folder <폴더경로> [출력폴더]")
        print("3. PDF 이미지: python image_flipper.py pdf [PDF이미지폴더]")
        exit()
    
    mode = sys.argv[1].lower()
    
    try:
        if mode == "single":
            if len(sys.argv) < 3:
                print("이미지 경로를 입력하세요")
                exit()
            
            image_path = sys.argv[2]
            result = flip_image_horizontal(image_path)
            print(f"완료: {result}")
            
        elif mode == "folder":
            if len(sys.argv) < 3:
                print("폴더 경로를 입력하세요")
                exit()
            
            input_folder = sys.argv[2]
            output_folder = sys.argv[3] if len(sys.argv) > 3 else None
            
            results = flip_folder_images(input_folder, output_folder)
            
        elif mode == "pdf":
            pdf_dir = sys.argv[2] if len(sys.argv) > 2 else "../output/PDF_Images"
            results = flip_pdf_images(pdf_dir)
            print(f"전체 처리 완료: {results['total_processed']}개 이미지")
            
        else:
            print("잘못된 모드입니다. single, folder, pdf 중 선택하세요")
            
    except Exception as e:
        print(f"오류 발생: {e}")