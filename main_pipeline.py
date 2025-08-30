# main_pipeline.py
def process_quest_request(image_data, mode, coords, stt_prompt):
    # Pipeline 1: SAM2 → SigLIP2 → 매뉴얼 매칭
    # Pipeline 2: ColPali → 매뉴얼 검색
    # LLM → 최종 응답
    pass