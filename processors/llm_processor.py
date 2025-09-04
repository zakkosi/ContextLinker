import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from PIL import Image
import io

class LLMProcessor:
    def __init__(self):
        """LLM 프로세서 초기화"""
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-5-nano"
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_bmw_manual_response(self, 
                               user_prompt: str,
                               manual_pages: List[Dict[str, Any]],
                               segmented_part: Optional[str] = None) -> Dict[str, Any]:
        try:
            # 시스템 프롬프트를 사용자 메시지에 포함
            full_prompt = f"""당신은 고도로 진보된 BMW 차량 전문가 AI 어시스턴트입니다. 
            다음 정보를 종합하여 사용자에게 가장 정확하고 유용한 답변을 제공하세요.

            사용자 질문: {user_prompt}
            
            답변 구조:
            - 부품/기능 명칭
            - 핵심 설명  
            - 사용 방법
            - 주의사항
            - 참조 정보
            """
            
            # input 데이터 구성
            content = [{"type": "input_text", "text": full_prompt}]
            
            # 매뉴얼 페이지 이미지들 추가
            if manual_pages:
                manual_info = f"\n관련 BMW 매뉴얼 페이지 {len(manual_pages)}개를 참조하세요:\n"
                
                for i, page in enumerate(manual_pages, 1):
                    try:
                        image_b64 = self.encode_image_to_base64(page['image_path'])
                        content.append({
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "auto"
                        })
                        manual_info += f"페이지 {i}: {page['image_name']} (유사도: {page['similarity_score']:.2f})\n"
                    except Exception as e:
                        print(f"매뉴얼 페이지 로딩 실패 {page['image_path']}: {e}")
                
                content[0]["text"] += manual_info
            
            # 세그먼트된 부품 이미지 추가
            if segmented_part and os.path.exists(segmented_part):
                try:
                    part_b64 = self.encode_image_to_base64(segmented_part)
                    content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{part_b64}",
                        "detail": "auto"
                    })
                    content[0]["text"] += "\n\n사용자가 가리킨 차량 부품 이미지도 함께 확인하세요."
                except Exception as e:
                    print(f"세그먼트 이미지 로딩 실패: {e}")
            
            # GPT-5 nano Responses API 호출
            input_data = [{
                "role": "user",
                "content": content
            }]
            
            response = self.client.responses.create(
                model=self.model,
                input=input_data,
                reasoning={"effort": "low"},
                text={"verbosity": "medium"}
            )
            
            return {
                'response': response.output[1].content[0].text,
                'model': self.model,
                'manual_pages_used': len(manual_pages),
                'has_segmented_part': segmented_part is not None
            }
            
        except Exception as e:
            print(f"LLM 처리 중 실제 오류: {e}")
            return {
                'error': f"LLM 응답 생성 실패: {e}",
                'response': "죄송합니다. 현재 BMW 매뉴얼 정보를 처리할 수 없습니다."
            }
    