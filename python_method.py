# test_gpt5_image.py
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_gpt5_nano_image():
    try:
        image_path = "test.png"  # 실제 이미지 파일명으로 변경
        encoded_image = encode_image(image_path)
        
        input_image = {
            "type": "input_image", 
            "image_url": f"data:image/jpeg;base64,{encoded_image}", 
            "detail": "auto"
        }
        
        input_data = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "이 이미지에서 무엇을 볼 수 있는지 설명해주세요."},
                    input_image,
                ],
            }
        ]
        
        response = client.responses.create(
            model="gpt-5-nano",
            input=input_data,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"}
        )
        
        print("GPT-5 nano 이미지 분석:")
        print(response.output[1].content[0].text)
        
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    test_gpt5_nano_image()