import torch
import numpy as np
from PIL import Image
import time
from typing import Tuple, Optional, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Segmenter:
    def __init__(self, checkpoint_path: str = "sam2/checkpoints/sam2.1_hiera_large.pt", 
                 model_cfg_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml"):
        """SAM2 세그멘터 초기화"""
        self.checkpoint_path = checkpoint_path
        self.model_cfg_path = model_cfg_path
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """모델 로딩"""
        print("SAM2 모델 로딩 중...")
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg_path, self.checkpoint_path))
        print("SAM2 모델 로딩 완료")
    
    def segment_with_point(self, image_object: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """모드 0: 중앙점 기반 세그먼트"""
        width, height = image_object.size
        input_point = np.array([[[width / 2, height / 2]]])
        input_label = np.array([[1]])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        return masks, scores
    
    def segment_with_bbox(self, image_object: Image.Image, bbox: list) -> Tuple[np.ndarray, np.ndarray]:
        """모드 1: 바운딩 박스 기반 세그먼트"""
        bbox_array = np.array(bbox)  # [x1, y1, x2, y2]
        
        masks, scores, _ = self.predictor.predict(
            box=bbox_array,
            multimask_output=True
        )
        return masks, scores
    
    def process_image(self, image_path: str, mode: int = 0, bbox: Optional[list] = None) -> Dict[str, Any]:
        """
        단일 이미지 처리
        
        Args:
            image_path: 이미지 파일 경로
            mode: 0 (중앙점) 또는 1 (바운딩 박스)
            bbox: [x1, y1, x2, y2] 형태의 바운딩 박스 좌표
            
        Returns:
            dict: {
                'segmented_image': np.ndarray,
                'mask': np.ndarray, 
                'score': float,
                'inference_time': float,
                'mode': int,
                'bbox': list or None
            }
        """
        image_object = Image.open(image_path).convert("RGB")
        image_array = np.array(image_object)
        
        start_time = time.time()
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_object)
            
            if mode == 0:
                masks, scores = self.segment_with_point(image_object)
            elif mode == 1:
                if bbox is None:
                    raise ValueError("모드 1에서는 바운딩 박스가 필요합니다")
                masks, scores = self.segment_with_bbox(image_object, bbox)
            else:
                raise ValueError("모드는 0 또는 1이어야 합니다")
        
        inference_time = time.time() - start_time
        
        # 최고 점수 마스크 선택
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # 세그먼트된 이미지 생성
        mask_3d = np.stack([best_mask] * 3, axis=-1)
        segmented_image = image_array * mask_3d
        
        return {
            'segmented_image': segmented_image,
            'mask': best_mask,
            'score': scores[best_mask_idx],
            'inference_time': inference_time,
            'mode': mode,
            'bbox': bbox
        }

    def process_single_request(self, image_path: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        FastAPI 요청 형태로 단일 이미지 처리
        
        Args:
            image_path: 이미지 파일 경로
            request_data: {"mode": 0 or 1, "bbox": [x1,y1,x2,y2] (optional)}
            
        Returns:
            처리 결과 딕셔너리
        """
        mode = request_data.get('mode', 0)
        bbox = request_data.get('bbox', None)
        
        return self.process_image(image_path, mode, bbox)
    
    def save_result(self, result: Dict[str, Any], output_path: str) -> str:
        """결과 이미지 저장"""
        segmented_pil = Image.fromarray(result['segmented_image'].astype(np.uint8))
        segmented_pil.save(output_path)
        return output_path

def batch_process_with_json(test_dir: str = "TEST", 
                           output_dir: str = "OUTPUT_FINAL", 
                           json_file: str = "annotations.json") -> list:
    """JSON 파일 기반 배치 처리 (기존 함수 유지)"""
    segmenter = SAM2Segmenter()
    
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 파일 로드
    with open(json_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"처리할 이미지: {len(annotations)}개")
    
    # 배치 처리
    total_time = 0
    results = []
    
    for filename, data in annotations.items():
        print(f"처리 중: {filename}")
        
        try:
            image_path = os.path.join(test_dir, filename)
            
            # 새로운 모듈 방식 사용
            result = segmenter.process_single_request(image_path, data)
            
            # 결과 저장
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_segmented.png")
            segmenter.save_result(result, output_path)
            
            total_time += result['inference_time']
            results.append({
                'file': filename,
                'mode': result['mode'],
                'time': result['inference_time'],
                'score': result['score'],
                'output': output_path
            })
            
            print(f"  완료: {result['inference_time']:.3f}초, 점수: {result['score']:.3f}")
            
        except Exception as e:
            print(f"  에러: {e}")
    
    # 결과 요약
    print(f"\n=== 처리 완료 ===")
    print(f"총 처리 시간: {total_time:.3f}초")
    print(f"평균 처리 시간: {total_time/len(results):.3f}초")
    print(f"결과 저장 폴더: {output_dir}")
    
    return results

# main.py에서 사용할 수 있는 간단한 인터페이스
def quick_segment(image_path: str, mode: int = 0, bbox: Optional[list] = None) -> Dict[str, Any]:
    """
    빠른 세그멘테이션 - main.py용 인터페이스
    
    Usage:
        # 중앙점 모드
        result = quick_segment("image.jpg", mode=0)
        
        # 바운딩 박스 모드  
        result = quick_segment("image.jpg", mode=1, bbox=[400, 110, 846, 571])
    """
    segmenter = SAM2Segmenter()
    return segmenter.process_image(image_path, mode, bbox)

if __name__ == "__main__":
    # 테스트 실행
    results = batch_process_with_json()