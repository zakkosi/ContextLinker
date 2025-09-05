import re
from typing import Dict, Optional

class PlaceholderProcessor:
    def __init__(self):
        """Placeholder 프로세서 초기화"""
        self.placeholder_pattern = re.compile(r'\{([^}]+)\}')
        
        # 지시어 매핑
        self.reference_words = {
            'default': '여기',
            'alternatives': ['이거', '저거', '이것', '저것', '해당 부분']
        }
        
        print("PlaceholderProcessor 초기화 완료")
    
    def process_text(self, text: str, input_type: str = "text") -> str:
        """
        입력 타입에 따라 텍스트 처리
        
        Args:
            text: 원본 텍스트 (placeholder 포함 가능)
            input_type: "text", "image", "text_image"
            
        Returns:
            str: 처리된 텍스트
        """
        if input_type == "text":
            # 텍스트만: placeholder 제거하고 원래 단어로 복원
            return self._restore_original_text(text)
            
        elif input_type == "image":
            # 이미지만: 텍스트 없음
            return ""
            
        elif input_type == "text_image":
            # 텍스트+이미지: placeholder를 지시어로 교체
            return self._replace_with_reference_words(text)
            
        else:
            raise ValueError(f"지원하지 않는 input_type: {input_type}")
    
    def _restore_original_text(self, text: str) -> str:
        """
        placeholder를 제거하고 원래 단어로 복원
        
        예: "{분할 스크린}에서 설정하고..." → "분할 스크린에서 설정하고..."
        
        Args:
            text: placeholder 포함된 텍스트
            
        Returns:
            str: 원본 단어로 복원된 텍스트
        """
        def replace_placeholder(match):
            return match.group(1)  # 중괄호 안의 내용만 반환
        
        processed_text = self.placeholder_pattern.sub(replace_placeholder, text)
        print(f"텍스트 전용 처리: '{text}' → '{processed_text}'")
        return processed_text
    
    def _replace_with_reference_words(self, text: str) -> str:
        """
        placeholder를 지시어로 교체
        
        예: "{분할 스크린}에서 설정하고..." → "여기서 설정하고..."
        
        Args:
            text: placeholder 포함된 텍스트
            
        Returns:
            str: 지시어로 교체된 텍스트
        """
        def replace_placeholder(match):
            return self.reference_words['default']  # "여기"로 교체
        
        processed_text = self.placeholder_pattern.sub(replace_placeholder, text)
        print(f"텍스트+이미지 처리: '{text}' → '{processed_text}'")
        return processed_text
    
    def extract_placeholders(self, text: str) -> list:
        """
        텍스트에서 placeholder 내용 추출
        
        Args:
            text: 원본 텍스트
            
        Returns:
            list: placeholder 내용 리스트
        """
        return self.placeholder_pattern.findall(text)
    
    def has_placeholders(self, text: str) -> bool:
        """
        텍스트에 placeholder가 있는지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            bool: placeholder 존재 여부
        """
        return bool(self.placeholder_pattern.search(text))

def test_placeholder_processor():
    """Placeholder 프로세서 테스트"""
    processor = PlaceholderProcessor()
    
    # 테스트 케이스들
    test_cases = [
        {
            "text": "{분할 스크린}에서 설정하고 표시할 수 있는 정보의 종류에는 어떤 것들이 있습니까?",
            "input_type": "text",
            "expected": "분할 스크린에서 설정하고 표시할 수 있는 정보의 종류에는 어떤 것들이 있습니까?"
        },
        {
            "text": "{분할 스크린}에서 설정하고 표시할 수 있는 정보의 종류에는 어떤 것들이 있습니까?",
            "input_type": "text_image",
            "expected": "여기서 설정하고 표시할 수 있는 정보의 종류에는 어떤 것들이 있습니까?"
        },
        {
            "text": "{컨트롤 디스플레이}에서 터치 스크린을 이용하여 지도를 어떻게 조작할 수 있습니까?",
            "input_type": "text",
            "expected": "컨트롤 디스플레이에서 터치 스크린을 이용하여 지도를 어떻게 조작할 수 있습니까?"
        },
        {
            "text": "{컨트롤 디스플레이}에서 터치 스크린을 이용하여 지도를 어떻게 조작할 수 있습니까?",
            "input_type": "text_image",
            "expected": "여기서 터치 스크린을 이용하여 지도를 어떻게 조작할 수 있습니까?"
        },
        {
            "text": "일반 텍스트 질문입니다.",
            "input_type": "text",
            "expected": "일반 텍스트 질문입니다."
        },
        {
            "text": "placeholder가 없는 질문",
            "input_type": "text_image", 
            "expected": "placeholder가 없는 질문"
        }
    ]
    
    print("=== Placeholder Processor 테스트 ===")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i} ---")
        print(f"입력: {case['text']}")
        print(f"모드: {case['input_type']}")
        print(f"기대: {case['expected']}")
        
        try:
            result = processor.process_text(
                text=case['text'],
                input_type=case['input_type']
            )
            print(f"결과: {result}")
            
            # 결과 검증
            if result == case['expected']:
                print("✅ 성공")
            else:
                print("❌ 실패")
                
        except Exception as e:
            print(f"오류: {e}")
    
    # placeholder 검출 테스트
    print(f"\n=== Placeholder 검출 테스트 ===")
    test_texts = [
        "{분할 스크린}에서 설정하고...",
        "일반 텍스트",
        "{트렁크}를 {리모컨}으로 열기"
    ]
    
    for text in test_texts:
        has_ph = processor.has_placeholders(text)
        placeholders = processor.extract_placeholders(text)
        print(f"텍스트: '{text}'")
        print(f"Placeholder 있음: {has_ph}")
        print(f"추출된 내용: {placeholders}")
        print()

if __name__ == "__main__":
    test_placeholder_processor()