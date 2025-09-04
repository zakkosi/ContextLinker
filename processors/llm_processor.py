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
    
    def convert_stt_to_search_query(self, stt_prompt: str) -> str:
        """
        STT 프롬프트를 ColPali 검색용 쿼리로 변환
        
        Args:
            stt_prompt: 원본 STT 프롬프트
            
        Returns:
            str: 최적화된 검색 쿼리
        """
        try:
            conversion_prompt = f"""
            다음 사용자의 음성 명령을 BMW 매뉴얼 검색에 적합한 키워드로 변환해주세요.
            
            사용자 음성: "{stt_prompt}"
            
            변환 규칙:
            1. 핵심 키워드만 추출
            2. BMW 매뉴얼에서 찾을 수 있는 용어로 변환
            3. 영어 키워드 우선 사용
            4. 2-5단어 내외로 간결하게
            
            예시:
            "이게 뭐야?" → "steering wheel"
            "트렁크 어떻게 닫아?" → "trunk close operation"
            "에어컨 켜는 법" → "air conditioning control"
            
            변환된 검색 쿼리만 반환하세요 (설명 없이):
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 간단한 변환이므로 mini 사용
                messages=[{"role": "user", "content": conversion_prompt}],
                max_tokens=50,
                temperature=0.3
            )
            
            search_query = response.choices[0].message.content.strip()
            return search_query
            
        except Exception as e:
            print(f"STT 변환 실패: {e}")
            return stt_prompt  # 실패시 원본 반환

# main.py용 간단한 인터페이스
def quick_manual_response(user_prompt: str, manual_pages: List[Dict], segmented_part: str = None) -> str:
    """
    빠른 매뉴얼 응답 생성 - main.py용 인터페이스
    
    Usage:
        response = quick_manual_response("이게 뭐야?", manual_pages, "segmented_wheel.png")
    """
    processor = LLMProcessor()
    result = processor.generate_bmw_manual_response(user_prompt, manual_pages, segmented_part)
    return result.get('response', result.get('error', '응답 생성 실패'))

def quick_query_conversion(stt_prompt: str) -> str:
    """
    빠른 STT → 검색쿼리 변환 - main.py용 인터페이스
    
    Usage:
        search_query = quick_query_conversion("트렁크 닫는 법 알려줘")
    """
    processor = LLMProcessor()
    return processor.convert_stt_to_search_query(stt_prompt)

if __name__ == "__main__":
    # 테스트
    processor = LLMProcessor()
    
    # STT → 검색쿼리 변환 테스트
    test_stt = "이게 뭐야?"
    search_query = processor.convert_stt_to_search_query(test_stt)
    print(f"STT: '{test_stt}' → 검색쿼리: '{search_query}'")
    
    # 더미 매뉴얼 페이지로 응답 테스트 (실제 이미지 있을 때)
    # manual_pages = [{'image_path': 'manual_page1.jpg', 'image_name': 'page1.jpg', 'similarity_score': 12.5}]
    # response = processor.generate_bmw_manual_response("이게 뭐야?", manual_pages)
    # print(f"LLM 응답: {response['response']}")