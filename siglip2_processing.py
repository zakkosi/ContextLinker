import torch
import numpy as np
from PIL import Image
import time
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F

class SigLIP2Processor:
    def __init__(self, model_id: str = "google/siglip2-so400m-patch16-naflex"):
        """SigLIP2 프로세서 초기화"""
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """모델과 프로세서 로딩"""
        print(f"SigLIP2 모델 로딩 중: {self.model_id}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            print("SigLIP2 모델 로딩 완료")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
    
    def encode_image(self, image_path: str) -> Tuple[torch.Tensor, float]:
        """
        이미지를 임베딩으로 변환
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            tuple: (image_embedding, processing_time)
        """
        image = Image.open(image_path).convert("RGB")
        
        start_time = time.time()
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(**inputs)
            # L2 정규화
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        processing_time = time.time() - start_time
        
        return image_features, processing_time
    
    def encode_text(self, text: str) -> Tuple[torch.Tensor, float]:
        """
        텍스트를 임베딩으로 변환
        
        Args:
            text: 입력 텍스트
            
        Returns:
            tuple: (text_embedding, processing_time)
        """
        start_time = time.time()
        
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_features = self.model.get_text_features(**inputs)
            # L2 정규화
            text_features = F.normalize(text_features, p=2, dim=-1)
        
        processing_time = time.time() - start_time
        
        return text_features, processing_time
    
    def compute_similarity(self, image_path: str, text: str) -> Dict[str, Any]:
        """
        이미지-텍스트 유사도 계산
        
        Args:
            image_path: 이미지 파일 경로
            text: 비교할 텍스트
            
        Returns:
            dict: 유사도 결과
        """
        # 이미지 인코딩
        image_features, img_time = self.encode_image(image_path)
        
        # 텍스트 인코딩
        text_features, txt_time = self.encode_text(text)
        
        # 코사인 유사도 계산
        similarity = torch.cosine_similarity(image_features, text_features)
        similarity_score = similarity.item()
        
        return {
            'image_path': image_path,
            'text': text,
            'similarity_score': similarity_score,
            'image_encoding_time': img_time,
            'text_encoding_time': txt_time,
            'total_time': img_time + txt_time
        }
    
    def batch_text_search(self, image_path: str, text_queries: List[str]) -> Dict[str, Any]:
        """
        하나의 이미지에 대해 여러 텍스트로 검색
        
        Args:
            image_path: 이미지 파일 경로
            text_queries: 텍스트 쿼리 리스트
            
        Returns:
            dict: 정렬된 유사도 결과
        """
        # 이미지는 한 번만 인코딩
        image_features, img_time = self.encode_image(image_path)
        
        results = []
        total_txt_time = 0
        
        for text in text_queries:
            text_features, txt_time = self.encode_text(text)
            total_txt_time += txt_time
            
            similarity = torch.cosine_similarity(image_features, text_features)
            similarity_score = similarity.item()
            
            results.append({
                'text': text,
                'similarity_score': similarity_score
            })
        
        # 유사도 순으로 정렬
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'image_path': image_path,
            'results': results,
            'image_encoding_time': img_time,
            'text_encoding_time': total_txt_time,
            'total_time': img_time + total_txt_time
        }
    
    def batch_image_search(self, image_paths: List[str], text_query: str) -> Dict[str, Any]:
        """
        하나의 텍스트에 대해 여러 이미지로 검색
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            text_query: 검색할 텍스트
            
        Returns:
            dict: 정렬된 유사도 결과
        """
        # 텍스트는 한 번만 인코딩
        text_features, txt_time = self.encode_text(text_query)
        
        results = []
        total_img_time = 0
        
        for image_path in image_paths:
            try:
                image_features, img_time = self.encode_image(image_path)
                total_img_time += img_time
                
                similarity = torch.cosine_similarity(image_features, text_features)
                similarity_score = similarity.item()
                
                results.append({
                    'image_path': image_path,
                    'similarity_score': similarity_score
                })
            except Exception as e:
                print(f"이미지 처리 실패 {image_path}: {e}")
        
        # 유사도 순으로 정렬
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'text_query': text_query,
            'results': results,
            'text_encoding_time': txt_time,
            'image_encoding_time': total_img_time,
            'total_time': txt_time + total_img_time
        }

def quick_similarity(image_path: str, text: str) -> float:
    """
    빠른 유사도 계산 - main.py용 간단 인터페이스
    
    Usage:
        score = quick_similarity("car.jpg", "steering wheel")
    """
    processor = SigLIP2Processor()
    result = processor.compute_similarity(image_path, text)
    return result['similarity_score']

def quick_text_search(image_path: str, text_queries: List[str]) -> List[Dict[str, Any]]:
    """
    빠른 텍스트 검색 - main.py용 인터페이스
    
    Usage:
        results = quick_text_search("car.jpg", ["steering wheel", "dashboard", "seat"])
    """
    processor = SigLIP2Processor()
    result = processor.batch_text_search(image_path, text_queries)
    return result['results']

if __name__ == "__main__":
    # 테스트 코드
    processor = SigLIP2Processor()
    
    # 단일 유사도 테스트
    result = processor.compute_similarity("TEST1.png", "steering wheel")
    print(f"유사도: {result['similarity_score']:.4f}")
    
    # 배치 텍스트 검색 테스트
    text_queries = ["steering wheel", "dashboard", "car interior", "seat", "trunk"]
    batch_result = processor.batch_text_search("TEST1.png", text_queries)
    
    print("\n텍스트 검색 결과:")
    for item in batch_result['results']:
        print(f"'{item['text']}': {item['similarity_score']:.4f}")