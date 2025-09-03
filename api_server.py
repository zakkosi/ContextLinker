import uvicorn
import os
import time
import tempfile
import json
import shutil
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# BMW Manual Assistant import
from main import BMWManualAssistant

app = FastAPI(title="BMW Manual Assistant API", version="1.0.1")

# --- CORS 설정 ---
origins = ["*"] # 테스트를 위해 모든 출처 허용 (실제 배포 시에는 특정 주소로 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 결과 파일 제공을 위한 설정 ---
OUTPUT_DIR = "temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- 글로벌 어시스턴트 인스턴스 ---
bmw_assistant: Optional[BMWManualAssistant] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 BMW Assistant 초기화"""
    global bmw_assistant
    print("BMW Manual Assistant 초기화 중...")
    try:
        bmw_assistant = BMWManualAssistant()
        print("BMW Manual Assistant 초기화 완료")
    except Exception as e:
        print(f"BMW Assistant 초기화 실패: {e}")
        bmw_assistant = None

@app.get("/", include_in_schema=False)
async def root():
    """테스트용 HTML 페이지 반환"""
    html_path = "static/index.html"
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Welcome to BMW Manual Assistant API. Place index.html in 'static' folder for web testing."}

@app.get("/health")
async def health_check():
    """헬스 체크"""
    if bmw_assistant is None:
        raise HTTPException(status_code=503, detail="BMW Assistant not initialized")
    return {"status": "healthy", "assistant_ready": True}

# --- 통합된 단일 엔드포인트 ---
@app.post("/quest")
async def handle_unified_quest(
    image: UploadFile = File(...),
    stt_prompt: str = Form(""),
    mode: int = Form(0),
    coords: Optional[str] = Form(None, description="JSON 문자열 형태의 좌표 [x1, y1, x2, y2]")
):
    """
    BMW 매뉴얼 질문을 처리하는 통합 엔드포인트
    - 이미지와 함께 질문(stt_prompt), 모드(mode), 좌표(coords)를 Form 데이터로 받습니다.
    """
    if bmw_assistant is None:
        raise HTTPException(status_code=503, detail="BMW Assistant not initialized")

    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Coords(좌표) 파싱
    bbox = None
    if coords and coords.strip():
        try:
            bbox = json.loads(coords)
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("좌표는 4개의 숫자를 가진 리스트여야 합니다.")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"잘못된 coords 형식입니다: {e}")

    tmp_image_path = None
    try:
        # 임시 파일로 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            shutil.copyfileobj(image.file, tmp_file)
            tmp_image_path = tmp_file.name

        # 요청 데이터 구성 및 처리
        request_data = {
            "stt_text": stt_prompt,
            "image_path": tmp_image_path,
            "mode": mode,
            "bbox": bbox
        }
        print(f"통합 엔드포인트 요청 처리: {request_data}")
        result = bmw_assistant.process_request(request_data)

        if not result.get('success', False):
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": result.get('error', 'Processing failed'),
                    "processing_time": result.get('processing_time', 0)
                }
            )

        # 결과 이미지 URL 생성
        def get_public_url(local_path):
            if not local_path or not os.path.exists(local_path):
                return ""
            filename = os.path.basename(local_path)
            public_path = os.path.join(OUTPUT_DIR, filename)
            shutil.copy(local_path, public_path)
            return f"/outputs/{filename}"

        # 최종 응답 반환
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "response": result.get('final_response', ''),
                "processing_time": result.get('processing_time', 0),
                "segmented_image_url": get_public_url(result.get('segmented_image_path')),
                "processed_image_url": get_public_url(result.get('processed_image_path')),
                "top_manual_pages": result.get('top_manual_pages', []),
                "detailed_times": result.get('detailed_times', {}),
                "schrodinger_mode": result.get('schrodinger_mode', '')
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
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
    if bmw_assistant:
        bmw_assistant.cleanup_temp_files()

if __name__ == "__main__":
    print("BMW Manual Assistant FastAPI 서버 시작 (단일 엔드포인트 모드)...")
    print("웹 브라우저에서 http://127.0.0.1:8000 로 접속하여 테스트하세요.")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")