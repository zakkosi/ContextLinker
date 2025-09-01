import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import time

def split_pdf_to_images(pdf_path: str, 
                       output_base_dir: str = "../output/PDF_Images",
                       dpi: int = 150,
                       image_format: str = "png") -> Dict[str, any]:
    """
    PDF를 페이지별 이미지로 분할
    
    Args:
        pdf_path: PDF 파일 경로
        output_base_dir: 출력 기본 디렉토리
        dpi: 이미지 해상도 (기본 150)
        image_format: 이미지 포맷 ("png" 또는 "jpg")
        
    Returns:
        dict: 변환 결과 정보
    """
    start_time = time.time()
    
    # PDF 파일 확인
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    
    # PDF 파일명 추출 (확장자 제거)
    pdf_name = Path(pdf_path).stem
    
    # 출력 폴더 생성 (PDF명으로 하위 폴더 생성)
    output_dir = os.path.join(output_base_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"PDF 분할 시작: {pdf_path}")
    print(f"출력 폴더: {output_dir}")
    
    # PDF 열기
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        print(f"총 페이지 수: {total_pages}")
        
        converted_files = []
        failed_pages = []
        
        # 각 페이지를 이미지로 변환
        for page_num in range(total_pages):
            try:
                page = pdf_document[page_num]
                
                # 이미지로 렌더링 (DPI 설정)
                mat = fitz.Matrix(dpi/72, dpi/72)  # 72 DPI 기준으로 스케일링
                pix = page.get_pixmap(matrix=mat)
                
                # 출력 파일명 생성
                if image_format.lower() == "jpg":
                    output_filename = f"page_{page_num+1:03d}.jpg"
                else:
                    output_filename = f"page_{page_num+1:03d}.png"
                
                output_path = os.path.join(output_dir, output_filename)
                
                # 이미지 저장
                if image_format.lower() == "jpg":
                    pix.save(output_path, output="jpeg")
                else:
                    pix.save(output_path, output="png")
                
                converted_files.append({
                    'page_number': page_num + 1,
                    'output_path': output_path,
                    'filename': output_filename
                })
                
                print(f"페이지 {page_num+1}/{total_pages} 완료: {output_filename}")
                
            except Exception as e:
                failed_pages.append({
                    'page_number': page_num + 1,
                    'error': str(e)
                })
                print(f"페이지 {page_num+1} 변환 실패: {e}")
        
        # PDF 문서 닫기
        pdf_document.close()
        
        processing_time = time.time() - start_time
        
        # 결과 요약
        result = {
            'pdf_name': pdf_name,
            'pdf_path': pdf_path,
            'output_dir': output_dir,
            'total_pages': total_pages,
            'converted_pages': len(converted_files),
            'failed_pages': len(failed_pages),
            'converted_files': converted_files,
            'failed_pages': failed_pages,
            'processing_time': processing_time,
            'dpi': dpi,
            'image_format': image_format
        }
        
        print(f"\n=== PDF 분할 완료 ===")
        print(f"성공: {len(converted_files)}페이지")
        print(f"실패: {len(failed_pages)}페이지")
        print(f"처리 시간: {processing_time:.2f}초")
        print(f"출력 폴더: {output_dir}")
        
        return result
        
    except Exception as e:
        raise Exception(f"PDF 처리 중 오류 발생: {e}")

def batch_split_pdfs(pdf_folder: str, 
                    output_base_dir: str = "../output/PDF_Images",
                    dpi: int = 150,
                    image_format: str = "png") -> List[Dict[str, any]]:
    """
    폴더 내 모든 PDF를 배치 처리
    
    Args:
        pdf_folder: PDF 파일들이 있는 폴더
        output_base_dir: 출력 기본 디렉토리
        dpi: 이미지 해상도
        image_format: 이미지 포맷
        
    Returns:
        list: 각 PDF별 변환 결과
    """
    pdf_folder_path = Path(pdf_folder)
    if not pdf_folder_path.exists():
        raise FileNotFoundError(f"PDF 폴더를 찾을 수 없습니다: {pdf_folder}")
    
    # PDF 파일들 찾기
    pdf_files = list(pdf_folder_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_folder}")
        return []
    
    print(f"배치 처리할 PDF: {len(pdf_files)}개")
    
    results = []
    total_start_time = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] 처리 중: {pdf_path.name}")
        
        try:
            result = split_pdf_to_images(
                str(pdf_path), 
                output_base_dir, 
                dpi, 
                image_format
            )
            results.append(result)
        except Exception as e:
            print(f"PDF 처리 실패 {pdf_path.name}: {e}")
            results.append({
                'pdf_name': pdf_path.stem,
                'pdf_path': str(pdf_path),
                'error': str(e),
                'converted_pages': 0
            })
    
    total_time = time.time() - total_start_time
    
    # 전체 요약
    total_converted = sum(r.get('converted_pages', 0) for r in results)
    successful_pdfs = len([r for r in results if 'error' not in r])
    
    print(f"\n=== 배치 처리 완료 ===")
    print(f"처리된 PDF: {successful_pdfs}/{len(pdf_files)}")
    print(f"총 변환 페이지: {total_converted}개")
    print(f"총 처리 시간: {total_time:.2f}초")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python pdf_splitter.py <PDF파일경로>")
        print("예시: python pdf_splitter.py ../data/Manual_PDF.pdf")
        exit()
    
    pdf_path = sys.argv[1]
    
    try:
        result = split_pdf_to_images(pdf_path)
        print(f"\n변환 완료: {result['converted_pages']}페이지 → {result['output_dir']}")
    except Exception as e:
        print(f"오류 발생: {e}")