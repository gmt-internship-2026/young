"""예시 UI 서버 — 회사 키오스크 프로그램이 들어올 자리의 임시 대체물.

TODO(기획서 9장 №7·№8): 회사 프로그램(UI) 파일을 받으면 이 서버와 demo_ui/는
제거하고, event_sender.py 규격으로 이벤트만 전달한다.

회사 프로그램 연동 계약(이 서버가 시연하는 것):
- 이벤트(엔진→UI): /data 폴링 또는 event_output(udp) — move_left/right, select,
  go_back, go_home, fill_id_fields(이름·주민번호) 등 config classes 목록
- 음성 안내(UI→엔진): POST /announce {"text": "발급하기 버튼"} — 포커스 항목
  설명처럼 화면 구조를 아는 쪽(UI)이 문구를 만들어 엔진 TTS로 읽힌다
- 주민등록증 인식(UI→엔진): POST /ocr/start — 본인확인 화면 진입 시 요청,
  인식 성공 시 fill_id_fields 이벤트가 온다. POST /ocr/stop으로 중단
"""
import asyncio
import os

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

try:
    import psutil
except ImportError:  # psutil 미설치 환경(개발 PC)에서도 서버는 떠야 한다
    psutil = None

DEMO_UI_HTML = "demo_ui/index.html"
RECENT_EVENT_COUNT = 20


class AnnounceBody(BaseModel):
    text: str


def create_app(state, config):
    app = FastAPI(title="Gesture Kiosk Demo UI (예시 — 회사 프로그램 대체 예정)")
    index_path = os.path.join(config["root_dir"], DEMO_UI_HTML)
    stream_interval_sec = 1.0 / config["demo_ui"]["stream_fps"]
    jpeg_quality = config["demo_ui"]["jpeg_quality"]
    ocr_timeout_sec = config["ocr"]["timeout_sec"]

    @app.get("/")
    async def serve_index():
        return FileResponse(index_path)

    async def _stream():
        while True:
            frame = state.get_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ret:
                await asyncio.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            await asyncio.sleep(stream_interval_sec)

    @app.get("/video_feed")
    async def video_feed():
        return StreamingResponse(
            _stream(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/data")
    async def get_data():
        events = []
        for e in state.event_log[-RECENT_EVENT_COUNT:]:
            item = {"class_name": e.class_name, "conf": round(e.conf, 2), "ts_sec": e.ts_sec}
            if e.hand_side is not None:
                item["hand_side"] = e.hand_side
            if e.data is not None:
                item["data"] = e.data
            events.append(item)
        return {
            "stats": {
                "cpu": psutil.cpu_percent(interval=None) if psutil else 0.0,
                "memory": psutil.virtual_memory().percent if psutil else 0.0,
                "capture_fps": round(state.capture_fps, 1),
                "infer_fps": round(state.infer_fps, 1),
            },
            "status": {
                "is_user_locked": state.is_user_locked,
                "is_ocr_mode": state.is_ocr_mode_active(),
            },
            "classes": config["classes"],
            "events": events,
        }

    # ----- 회사 프로그램 연동 계약 엔드포인트 -----

    @app.post("/announce")
    async def announce(body: AnnounceBody):
        """UI가 화면 맥락(포커스 항목 등)을 음성 안내로 요청한다."""
        state.announcer.announce(body.text)
        return {"ok": True}

    @app.post("/ocr/start")
    async def ocr_start():
        """본인확인 화면 진입 — 주민등록증 인식 모드를 켠다."""
        state.start_ocr_mode(ocr_timeout_sec)
        return {"ok": True, "timeout_sec": ocr_timeout_sec}

    @app.post("/ocr/stop")
    async def ocr_stop():
        state.stop_ocr_mode()
        return {"ok": True}

    return app
