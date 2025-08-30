import os
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Dict, Any, Optional, Tuple
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

class ColPaliProcessor:
    def __init__(self, model_name: str = "tsystems/colqwen2.5-3b-multilingual-v1.0", device: str = "cuda:0"):
        """ColPali 프로세서 초기화"""
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.database = []
        self.db_embeddings = []
        self._load_model()
    
    def _load_model(self):
        """모델과 프로세서 로딩"""
        print(f"ColPali 모델 로딩 중: {self.model_name}")
        try:
            self.model = ColQwen2_5.from_pretrained(
                self.model_name, 
                torch_dtype=torch.bfloat16, 
                device_map=self.device
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
            print("ColPali 모델 로딩 완료")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
    
    def build_database_from_folder(self, db_folder_path: str, batch_size: int = 8) -> Tuple[int, float]:
        """
        폴더의 이미지들로 데이터베이스 구축
        
        Args:
            db_folder_path: 이미지 폴더 경로
            batch_size: 배치 처리 크기
            
        Returns:
            tuple: (처리된 이미지 수, 총 처리 시간)
        """
        db_path = Path(db_folder_path)
        if not db_path.exists():
            raise ValueError(f"폴더를 찾을 수 없습니다: {db_folder_path}")
        
        print(f"ColPali: 폴더에서 데이터베이스 구축 중: {db_path}")
        
        self.database = []
        self.db_embeddings = []
        
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [p for p in db_path.iterdir() if p.suffix.lower() in supported_extensions]
        
        if not image_files:
            print("지원되는 이미지 파일을 찾을 수 없습니다")
            return 0, 0.0
        
        print(f"ColPali: {len(image_files)}개 이미지 발견, 배치 크기 {batch_size}로 처리")
        
        start_time = time.time()
        
        with tqdm(total=len(image_files), desc="임베딩 생성") as pbar:
            for i in range(0, len(image_files), batch_size):
                batch_paths = image_files[i:i+batch_size]
                batch_images = []
                
                for image_path in batch_paths:
                    try:
                        pil_image = Image.open(image_path).convert("RGB")
                        item = {
                            'image_name': image_path.name,
                            'image_path': str(image_path),
                            'file_number': image_path.stem,
                            'pil_image': pil_image
                        }
                        self.database.append(item)
                        batch_images.append(pil_image)
                    except Exception as e:
                        print(f"이미지 처리 실패 {image_path}: {e}")
                
                if batch_images:
                    with torch.no_grad():
                        inputs = self.processor.process_images(images=batch_images).to(self.device)
                        batch_embeddings = list(torch.unbind(self.model(**inputs).cpu()))
                        self.db_embeddings.extend(batch_embeddings)
                
                pbar.update(len(batch_paths))
        
        total_time = time.time() - start_time
        print(f"ColPali: 데이터베이스 구축 완료 ({len(self.database)}개 이미지, {total_time:.3f}초)")
        
        return len(self.database), total_time
    
    def search_by_text(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """
        텍스트 쿼리로 이미지 검색
        
        Args:
            query_text: 검색할 텍스트
            k: 반환할 결과 수
            
        Returns:
            dict: 검색 결과
        """
        if not self.database:
            raise ValueError("데이터베이스가 구축되지 않았습니다. build_database_from_folder() 먼저 실행하세요")
        
        start_time = time.time()
        
        with torch.no_grad():
            # 텍스트 쿼리 임베딩
            inputs = self.processor.process_queries(queries=[query_text]).to(self.device)
            query_embed = list(torch.unbind(self.model(**inputs).cpu()))
            
            # 유사도 계산
            scores = self.processor.score(query_embed, self.db_embeddings, device=self.device)
            top_k = scores[0].topk(k)
            
            indices = top_k.indices.tolist()
            similarities = top_k.values.tolist()
        
        search_time = time.time() - start_time
        
        # 결과 구성
        results = []
        for idx, sim_score in zip(indices, similarities):
            if idx < len(self.database):
                results.append({
                    'image_name': self.database[idx]['image_name'],
                    'image_path': self.database[idx]['image_path'],
                    'similarity_score': sim_score,
                    'rank': len(results) + 1
                })
        
        return {
            'query': query_text,
            'results': results,
            'search_time': search_time,
            'total_images': len(self.database)
        }
    
    def search_single_image(self, image_path: str, query_text: str) -> Dict[str, Any]:
        """
        단일 이미지에 대한 텍스트 쿼리 점수 계산
        
        Args:
            image_path: 이미지 파일 경로
            query_text: 검색할 텍스트
            
        Returns:
            dict: 유사도 결과
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"이미지 로딩 실패: {e}")
        
        start_time = time.time()
        
        with torch.no_grad():
            # 이미지 임베딩
            img_inputs = self.processor.process_images(images=[image]).to(self.device)
            img_embedding = list(torch.unbind(self.model(**img_inputs).cpu()))
            
            # 텍스트 임베딩
            txt_inputs = self.processor.process_queries(queries=[query_text]).to(self.device)
            txt_embedding = list(torch.unbind(self.model(**txt_inputs).cpu()))
            
            # 유사도 계산
            scores = self.processor.score(txt_embedding, img_embedding, device=self.device)
            similarity_score = scores[0][0].item()
        
        processing_time = time.time() - start_time
        
        return {
            'image_path': image_path,
            'query_text': query_text,
            'similarity_score': similarity_score,
            'processing_time': processing_time
        }

# main.py용 간단한 인터페이스 함수들
def quick_colpali_search(db_folder: str, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    빠른 ColPali 검색 - main.py용 인터페이스
    
    Usage:
        results = quick_colpali_search("./images", "steering wheel", k=3)
    """
    processor = ColPaliProcessor()
    processor.build_database_from_folder(db_folder)
    result = processor.search_by_text(query_text, k)
    return result['results']

def quick_single_match(image_path: str, query_text: str) -> float:
    """
    단일 이미지-텍스트 유사도 - main.py용 인터페이스
    
    Usage:
        score = quick_single_match("car.jpg", "steering wheel")
    """
    processor = ColPaliProcessor()
    result = processor.search_single_image(image_path, query_text)
    return result['similarity_score']

def test_colpali(test_image_path: str, test_queries: List[str]):
    """ColPali 테스트 함수"""
    processor = ColPaliProcessor()
    
    print(f"테스트 이미지: {test_image_path}")
    print("=" * 50)
    
    for query in test_queries:
        result = processor.search_single_image(test_image_path, query)
        print(f"쿼리: '{query}'")
        print(f"유사도: {result['similarity_score']:.4f}")
        print(f"처리 시간: {result['processing_time']:.3f}초")
        print("-" * 30)

if __name__ == "__main__":
    # 테스트 실행
    test_queries = [
        "steering wheel",
        "dashboard", 
        "car interior",
        "BMW logo",
        "seat"
    ]
    
    test_colpali("TEST1.png", test_queries)