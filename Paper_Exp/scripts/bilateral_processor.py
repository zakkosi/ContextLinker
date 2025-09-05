import cv2
import numpy as np
from PIL import Image
import time
from typing import Optional
import os

class BilateralFilterProcessor:
    def __init__(self, d: int = 15, sigma_color: float = 80, sigma_space: float = 80):
        """
        Bilateral Filter 프로세서 초기화 (Schrodinger Bridge 대체)
        매뉴얼 스타일 처리: 그레이스케일 → Bilateral Filter → 외곽선 강조
        
        Args:
            d: 필터링 중 사용되는 각 픽셀 이웃의 지름 (클수록 더 부드러운 효과)
            sigma_color: 색상 차이에 대한 시그마 값 (클수록 더 많은 색상 혼합)
            sigma_space: 좌표 공간에서의 시그마 값 (클수록 더 넓은 영역 고려)
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        print(f"BilateralFilter 프로세서 초기화 완료 (d={d}, σ_color={sigma_color}, σ_space={sigma_space})")
    
    def manual_style_filter(self, image_path: str, output_dir: str = "temp") -> str:
        """
        실제 이미지 → 매뉴얼 스타일 변환 (Schrodinger 대체)
        그레이스케일 → Bilateral Filter → 외곽선 강조로 매뉴얼 같은 효과 생성
        
        Args:
            image_path: 입력 이미지 경로
            output_dir: 출력 디렉토리
            
        Returns:
            str: 처리된 이미지 파일 경로
        """
        try:
            # 입력 이미지 로드
            input_image = Image.open(image_path).convert("RGB")
            image_array = np.array(input_image)
            
            # 출력 경로 생성 (입력 파일명 기반)
            os.makedirs(output_dir, exist_ok=True)
            input_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"bilateral_filtered_{input_filename}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            start_time = time.time()
            
            # 1. RGB → BGR 변환 (OpenCV용)
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # 2. 그레이스케일 변환
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            
            # 3. Bilateral Filter 적용 (노이즈 제거하면서 엣지 보존)
            filtered_gray = cv2.bilateralFilter(gray_image, self.d, self.sigma_color, self.sigma_space)
            
            # 4. 외곽선 강조 처리 (매뉴얼 스타일)
            # 추가적인 엣지 강조를 위한 언샤프 마스킹
            blurred = cv2.GaussianBlur(filtered_gray, (5, 5), 0)
            sharpened = cv2.addWeighted(filtered_gray, 1.5, blurred, -0.5, 0)
            
            # 5. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            
            # 6. 3채널로 변환 (RGB 형태로)
            manual_style_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            manual_style_rgb = cv2.cvtColor(manual_style_bgr, cv2.COLOR_BGR2RGB)
            
            # 7. 결과 저장
            output_image = Image.fromarray(manual_style_rgb)
            output_image.save(output_path)
            
            processing_time = time.time() - start_time
            print(f"Bilateral 매뉴얼 스타일 변환 완료: {output_path} ({processing_time:.3f}초)")
            
            return output_path
            
        except Exception as e:
            print(f"Bilateral 필터링 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._placeholder_process(image_path, output_dir, "bilateral_filtered")
    
    def _placeholder_process(self, image_path: str, output_dir: str, mode: str) -> str:
        """
        모델 실패시 placeholder 처리
        
        Args:
            image_path: 입력 이미지 경로
            output_dir: 출력 디렉토리  
            mode: 처리 모드명
            
        Returns:
            str: placeholder 이미지 경로
        """
        try:
            input_image = Image.open(image_path).convert("RGB")
            
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{mode}_{int(time.time())}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            input_image.save(output_path)
            print(f"[PLACEHOLDER] {mode}: {image_path} → {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Placeholder 처리 실패: {e}")
            raise

def test_bilateral_processor():
    """Bilateral Filter 프로세서 테스트"""
    processor = BilateralFilterProcessor()
    
    # 현재 실행 위치에 따라 경로 자동 조정
    if os.path.exists("Paper_Exp/Query_images"):
        # 루트에서 실행하는 경우
        input_dir = "Paper_Exp/Query_images"
        output_dir = "Paper_Exp/output/bilateral_test_output"
    elif os.path.exists("../Query_images"):
        # scripts 폴더에서 실행하는 경우
        input_dir = "../Query_images"
        output_dir = "../output/bilateral_test_output"
    else:
        print("Query_images 폴더를 찾을 수 없습니다.")
        print("현재 경로:", os.getcwd())
        return
    
    # 테스트할 이미지들 (1.png ~ 6.png)
    test_images = [f"{i}.png" for i in range(1, 7)]
    
    print(f"현재 실행 디렉토리: {os.getcwd()}")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in test_images:
        img_path = os.path.join(input_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"이미지를 찾을 수 없습니다: {img_path}")
            continue
            
        try:
            print(f"\n=== {img_name} 처리 중 ===")
            
            # 기본 파라미터로 테스트
            result_path = processor.manual_style_filter(img_path, output_dir)
            print(f"기본 필터 완료: {result_path}")
            
        except Exception as e:
            print(f"{img_name} 처리 중 오류: {e}")
    
    print(f"\n모든 테스트 완료. 결과는 {output_dir} 폴더에 저장됨.")

if __name__ == "__main__":
    test_bilateral_processor()