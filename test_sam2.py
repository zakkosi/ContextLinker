import torch
import numpy as np
from PIL import Image
import time
import os
import json

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def segment_with_point(predictor, image_object):
    """모드 0: 중앙점 기반 세그먼트"""
    width, height = image_object.size
    input_point = np.array([[[width / 2, height / 2]]])
    input_label = np.array([[1]])
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    return masks, scores

def segment_with_bbox(predictor, image_object, bbox):
    """모드 1: 바운딩 박스 기반 세그먼트"""
    bbox_array = np.array(bbox)  # [x1, y1, x2, y2]
    
    masks, scores, _ = predictor.predict(
        box=bbox_array,
        multimask_output=True
    )
    return masks, scores

def process_single_image(predictor, image_path, mode=0, bbox=None):
    """단일 이미지 처리"""
    image_object = Image.open(image_path).convert("RGB")
    image_array = np.array(image_object)
    
    start_time = time.time()
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_object)
        
        if mode == 0:
            masks, scores = segment_with_point(predictor, image_object)
        elif mode == 1:
            if bbox is None:
                raise ValueError("모드 1에서는 바운딩 박스가 필요합니다")
            masks, scores = segment_with_bbox(predictor, image_object, bbox)
        else:
            raise ValueError("모드는 0 또는 1이어야 합니다")
    
    inference_time = time.time() - start_time
    
    # 최고 점수 마스크 선택
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    
    # 세그먼트된 이미지 생성
    mask_3d = np.stack([best_mask] * 3, axis=-1)
    segmented_image = image_array * mask_3d
    
    return segmented_image, best_mask, scores[best_mask_idx], inference_time

def batch_process_with_json(test_dir="TEST", output_dir="OUTPUT_FINAL", json_file="annotations.json"):
    """JSON 파일 기반 배치 처리"""
    # 설정
    CHECKPOINT_PATH = "sam2/checkpoints/sam2.1_hiera_large.pt"
    MODEL_CFG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로딩
    print("SAM2 모델 로딩 중...")
    predictor = SAM2ImagePredictor(build_sam2(MODEL_CFG_PATH, CHECKPOINT_PATH))
    print("모델 로딩 완료")
    
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
            mode = data['mode']
            bbox = data.get('bbox', None)
            
            # 이미지 처리
            segmented_image, mask, score, inference_time = process_single_image(
                predictor, image_path, mode, bbox
            )
            
            # 결과 저장
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_segmented.png")
            
            segmented_pil = Image.fromarray(segmented_image.astype(np.uint8))
            segmented_pil.save(output_path)
            
            total_time += inference_time
            results.append({
                'file': filename,
                'mode': mode,
                'time': inference_time,
                'score': score,
                'output': output_path
            })
            
            print(f"  완료: {inference_time:.3f}초, 점수: {score:.3f}")
            
        except Exception as e:
            print(f"  에러: {e}")
    
    # 결과 요약
    print(f"\n=== 처리 완료 ===")
    print(f"총 처리 시간: {total_time:.3f}초")
    print(f"평균 처리 시간: {total_time/len(results):.3f}초")
    print(f"결과 저장 폴더: {output_dir}")
    
    return results

if __name__ == "__main__":
    # 테스트 실행
    results = batch_process_with_json()