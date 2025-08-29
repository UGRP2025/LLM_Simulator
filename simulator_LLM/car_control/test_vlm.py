# test_vlm_image_png.py
import base64
import cv2
import sys
from openai import OpenAI

# --- 설정 ---
API_BASE = "http://127.0.0.1:8000/v1"   # vLLM 서버 주소
API_KEY  = "EMPTY"                       # 형식상 필요. vLLM은 보통 검증 안 함
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" # /v1/models에서 보이는 id로 맞추기
IMG_PATH = "racing.png"                  # 테스트 이미지 경로

def main():
    # 이미지 로드
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERR] 이미지를 못 읽었습니다: {IMG_PATH}")
        sys.exit(1)

    # (선택) 크기 축소: 지연시간 절약
    img = cv2.resize(img, (640, 384), interpolation=cv2.INTER_AREA)

    # PNG 인코딩 (압축 레벨 3 정도가 속도/용량 균형)
    ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        print("[ERR] PNG 인코딩 실패")
        sys.exit(1)

    # base64 + data URL
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    # OpenAI 호환 클라이언트
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    # 멀티모달 요청
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,  # 서버에 등록된 모델 id와 동일해야 함
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=64,
            temperature=0.1,
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        # 서버 4xx/5xx 등 에러 내용 확인
        print(f"[ERR] request failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
