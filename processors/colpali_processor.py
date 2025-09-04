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
        self.db_embeddings = []
        self.db_image_paths = []
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
    
    def build_database_from_pdf_images(self, pdf_images_folder: str, batch_size: int = 8) -> Tuple[int, float]:
        """
        PDF 이미지 폴더들로 데이터베이스 구축 (중첩 폴더 구조 지원)
        
        Args:
            pdf_images_folder: PDF 이미지들이 있는 루트 폴더 (예: "output/PDF_Images")
            batch_size: 배치 처리 크기
            
        Returns:
            tuple: (처리된 이미지 수, 총 처리 시간)
        """
        pdf_path = Path(pdf_images_folder)
        if not pdf_path.exists():
            raise ValueError(f"PDF 이미지 폴더를 찾을 수 없습니다: {pdf_images_folder}")
        
        print(f"PDF 이미지 데이터베이스 구축 중: {pdf_path}")
        
        # 초기화
        self.db_embeddings = []
        self.db_image_paths = []
        
        # 모든 PDF 이미지 파일 수집
        all_images = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for pdf_folder in pdf_path.iterdir():
            if pdf_folder.is_dir():
                for ext in supported_extensions:
                    all_images.extend(list(pdf_folder.glob(f"*{ext}")))
                    all_images.extend(list(pdf_folder.glob(f"*{ext.upper()}")))
        
        if not all_images:
            print("PDF 이미지를 찾을 수 없습니다")
            return 0, 0.0
        
        print(f"총 {len(all_images)}개 PDF 이미지 발견, 배치 크기 {batch_size}로 처리")
        
        start_time = time.time()
        
        with torch.no_grad():
            with tqdm(total=len(all_images), desc="PDF 이미지 임베딩 생성") as pbar:
                for i in range(0, len(all_images), batch_size):
                    batch_paths = all_images[i:i+batch_size]
                    batch_images = []
                    valid_paths = []
                    
                    # 이미지 로드
                    for img_path in batch_paths:
                        try:
                            pil_image = Image.open(img_path).convert("RGB")
                            batch_images.append(pil_image)
                            valid_paths.append(str(img_path))
                        except Exception as e:
                            print(f"이미지 로드 실패 {img_path}: {e}")
                    
                    if batch_images:
                        # ColPali로 임베딩 생성
                        inputs = self.processor.process_images(images=batch_images).to(self.device)
                        batch_embeddings = list(torch.unbind(self.model(**inputs).cpu()))
                        
                        self.db_embeddings.extend(batch_embeddings)
                        self.db_image_paths.extend(valid_paths)
                    
                    pbar.update(len(batch_paths))
        
        total_time = time.time() - start_time
        print(f"PDF 이미지 데이터베이스 구축 완료 ({len(self.db_embeddings)}개 이미지, {total_time:.3f}초)")
        
        return len(self.db_embeddings), total_time
    
    def search_by_image(self, query_image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        이미지로 유사한 이미지들 검색
        
        Args:
            query_image_path: 쿼리 이미지 파일 경로
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        if not self.db_embeddings:
            raise ValueError("데이터베이스가 구축되지 않았습니다. build_database_from_pdf_images() 먼저 실행하세요")
        
        try:
            # 쿼리 이미지 임베딩
            query_image = Image.open(query_image_path).convert("RGB")
            
            with torch.no_grad():
                query_inputs = self.processor.process_images(images=[query_image]).to(self.device)
                query_embedding = list(torch.unbind(self.model(**query_inputs).cpu()))
                
                # 유사도 계산
                scores = self.processor.score(query_embedding, self.db_embeddings, device=self.device)
                top_k = scores[0].topk(k)
                
                indices = top_k.indices.tolist()
                similarities = top_k.values.tolist()
            
            # 결과 구성
            results = []
            for idx, sim_score in zip(indices, similarities):
                if idx < len(self.db_image_paths):
                    image_path = self.db_image_paths[idx]
                    results.append({
                        'image_name': Path(image_path).name,
                        'image_path': image_path,
                        'similarity_score': sim_score,
                        'rank': len(results) + 1
                    })
            
            return results
            
        except Exception as e:
            print(f"이미지 검색 실패: {e}")
            return []
    
    def search_by_text(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        텍스트 쿼리로 이미지 검색
        
        Args:
            query_text: 검색할 텍스트
            k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        if not self.db_embeddings:
            raise ValueError("데이터베이스가 구축되지 않았습니다. build_database_from_pdf_images() 먼저 실행하세요")
        
        start_time = time.time()
        
        with torch.no_grad():
            # 텍스트 쿼리 임베딩
            inputs = self.processor.process_queries(queries=[query_text]).to(self.device)
            query_embed = self.model(**inputs)
            
            # 유사도 계산 - score_multi_vector 사용
            scores = self.processor.score_multi_vector(query_embed, torch.stack(self.db_embeddings))
            top_k = scores[0].topk(k)
            
            indices = top_k.indices.tolist()
            similarities = top_k.values.tolist()
        
        search_time = time.time() - start_time
        
        # 결과 구성
        results = []
        for idx, sim_score in zip(indices, similarities):
            if idx < len(self.db_image_paths):
                image_path = self.db_image_paths[idx]
                results.append({
                    'image_name': Path(image_path).name,
                    'image_path': image_path,
                    'similarity_score': sim_score,
                    'rank': len(results) + 1
                })
        
        return results
    
    def search_single_image(self, image_path: str, query_text: str) -> Dict[str, Any]:
        """
        단일 이미지에 대한 텍스트 쿼리 점수 계산
        
        Args:
            image_path: 이미지 파일 경로
            query_text: 검색할 텍스트
            
        Returns:
            Dict: 유사도 결과
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

    def get_database_info(self) -> Dict[str, Any]:
        """데이터베이스 정보 반환"""
        return {
            'total_images': len(self.db_embeddings),
            'image_paths': self.db_image_paths,
            'model_name': self.model_name,
            'device': self.device
        }
    
if __name__ == "__main__":
    # 테스트 실행
    processor = ColPaliProcessor()
    
    # 데이터베이스 구축 테스트
    try:
        num_images, build_time = processor.build_database_from_pdf_images("output/PDF_Images")
        print(f"데이터베이스 구축 완료: {num_images}개 이미지, {build_time:.2f}초")
        
        # 데이터베이스 정보 출력
        info = processor.get_database_info()
        print(f"데이터베이스 정보: {info}")
        
    except Exception as e:
        print(f"테스트 실패: {e}")