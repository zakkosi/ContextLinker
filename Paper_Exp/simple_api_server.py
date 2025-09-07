import uvicorn
import os
import sys
import time
import tempfile
import json
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 프로젝트 루트 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

# 단순화된 파이프라인 import
from Paper_Exp.scripts.simple_experiment_pipeline import SimpleExperimentPipeline
from Paper_Exp.scripts.placeholder_processor import PlaceholderProcessor

app = FastAPI(title="단순화된 BMW Manual Assistant API", version="2.0.0")

# --- CORS 설정 ---
origins = ["*"] # 테스트를 위해 모든 출처 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 결과 파일 제공을 위한 설정 ---
OUTPUT_DIR = "Paper_Exp/temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- 글로벌 파이프라인 인스턴스 ---
simple_pipeline: Optional[SimpleExperimentPipeline] = None
placeholder_processor: Optional[PlaceholderProcessor] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 단순화된 파이프라인 초기화"""
    global simple_pipeline, placeholder_processor
    print("단순화된 BMW Manual Assistant 초기화 중...")
    try:
        # 작업 디렉토리 변경
        os.chdir(PROJECT_ROOT)
        
        # 파이프라인 초기화
        simple_pipeline = SimpleExperimentPipeline()
        placeholder_processor = PlaceholderProcessor()
        
        print("단순화된 BMW Manual Assistant 초기화 완료")
    except Exception as e:
        print(f"파이프라인 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        simple_pipeline = None
        placeholder_processor = None

@app.get("/", include_in_schema=False)
async def root():
    """테스트용 HTML 페이지 반환"""
    html_path = "Paper_Exp/static/simple_index.html"
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "단순화된 BMW Manual Assistant API. Text/Text+Image 모드 지원."}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    if simple_pipeline is None or placeholder_processor is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {"status": "healthy", "pipeline_ready": True, "mode": "simplified"}

@app.post("/simple_quest")
async def handle_simple_quest(
    text_query: str = Form(..., description="텍스트 질문"),
    image: Optional[UploadFile] = File(None, description="선택적 이미지 (Text+Image 모드)")
):
    """
    단순화된 BMW 매뉴얼 질문 처리 엔드포인트
    - text_query: 필수 텍스트 질문
    - image: 선택적 이미지 (있으면 Text+Image 모드, 없으면 Text 모드)
    """
    if simple_pipeline is None or placeholder_processor is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start_time = time.time()
    tmp_image_path = None
    
    try:
        # 입력 타입 결정
        if image and image.content_type.startswith('image/'):
            input_type = "text_image"
            # 임시 파일로 이미지 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                shutil.copyfileobj(image.file, tmp_file)
                tmp_image_path = tmp_file.name
        else:
            input_type = "text"
            tmp_image_path = None

        print(f"단순화된 API 요청 처리: {input_type} 모드")
        print(f"텍스트: {text_query}")
        if tmp_image_path:
            print(f"이미지: {tmp_image_path}")

        # 1. 텍스트 처리 (placeholder 변환)
        processed_text = placeholder_processor.process_text(text_query, input_type)
        print(f"처리된 텍스트: '{processed_text}'")

        # 2. ColPali 검색
        if input_type == "text":
            # 텍스트만으로 검색
            search_results = simple_pipeline.colpali_processor.search_by_text(processed_text, k=5)
        else:
            # 이미지로 검색
            search_results = simple_pipeline.colpali_processor.search_by_image(tmp_image_path, k=5)

        print(f"검색된 페이지 수: {len(search_results)}")

        # 3. LLM 답변 생성
        llm_response = simple_pipeline.llm_processor.generate_bmw_manual_response(
            user_prompt=processed_text,
            manual_pages=search_results,
            segmented_part=tmp_image_path if input_type == "text_image" else None
        )

        total_time = time.time() - start_time

        # 4. 검색된 매뉴얼 페이지 정보 구성
        top_manual_pages = []
        for result in search_results:
            top_manual_pages.append({
                "page_name": result.get('image_name', ''),
                "similarity_score": float(result.get('similarity_score', 0)),
                "rank": result.get('rank', 0)
            })

        # 5. 최종 응답 반환
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "mode": input_type,
                "original_query": text_query,
                "processed_text": processed_text,
                "response": llm_response.get('response', ''),
                "top_manual_pages": top_manual_pages,
                "processing_time": round(total_time, 2),
                "has_image": tmp_image_path is not None
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "processing_time": round(time.time() - start_time, 2)
            }
        )
    finally:
        # 임시 업로드 파일 정리
        if tmp_image_path and os.path.exists(tmp_image_path):
            os.unlink(tmp_image_path)

@app.on_event("shutdown")
def shutdown_event():
    """서버 종료 시 임시 폴더 정리"""
    print("임시 출력 폴더 정리 중...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

if __name__ == "__main__":
    print("단순화된 BMW Manual Assistant FastAPI 서버 시작...")
    print("지원 모드: Text only, Text+Image")
    print("웹 브라우저에서 http://127.0.0.1:8000 로 접속하여 테스트하세요.")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")