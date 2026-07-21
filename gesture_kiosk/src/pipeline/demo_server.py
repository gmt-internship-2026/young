"""예시 UI 서버 — 회사 키오스크 프로그램이 들어올 자리의 임시 대체물.

TODO(기획서 9장 №7·№8): 회사 프로그램(UI) 파일을 받으면 이 서버와 demo_ui/는
제거하고, event_sender.py 규격으로 이벤트만 전달한다.

회사 프로그램 연동 계약(이 서버가 시연하는 것):
- 이벤트(엔진→UI): /data 폴링 또는 event_output(udp) — move_left/right, select,
  go_back, go_home (config classes 목록)
(2026-07-16: 주민등록증 OCR 기능 제거 — 제스처 집중, /ocr/* 엔드포인트 삭제)
"""
import asyncio
import os

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

try:
    import psutil
except ImportError:  # psutil 미설치 환경(개발 PC)에서도 서버는 떠야 한다
    psutil = None

DEMO_UI_HTML = "demo_ui/index.html"
RECENT_EVENT_COUNT = 20


def create_app(state, config):
    app = FastAPI(title="Gesture Kiosk Demo UI (예시 — 회사 프로그램 대체 예정)")
    index_path = os.path.join(config["root_dir"], DEMO_UI_HTML)
    stream_interval_sec = 1.0 / config["demo_ui"]["stream_fps"]
    jpeg_quality = config["demo_ui"]["jpeg_quality"]

    @app.get("/")
    async def serve_index():
        return FileResponse(index_path)

    async def _stream():
        # 시청자 등록 — 파이프라인이 오버레이 렌더링을 켠다 (0명이면 그리기 생략,
        # 2026-07-20 최적화). 클라이언트가 끊기면 제너레이터 종료로 finally가 돈다
        state.add_viewer()
        try:
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
        finally:
            state.remove_viewer()

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
            },
            "debug": state.debug,   # 판정 계기판 — 실기 튜닝용 (연동 계약 아님)
            "classes": config["classes"],
            "events": events,
        }

    # ----- 회사 프로그램 연동 계약 엔드포인트 -----

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        """이벤트 실시간 push(2026-07-21 웹소켓 전환 — 회사 결정).

        접속해 두면 이벤트 확정 순간 UDP와 동일한 텍스트 한 줄이 온다:
            GESTURE|select|left|1.00|12345.678
        델파이 수신부는 UDP 파서를 그대로 재사용하면 된다 (docs/델파이7_연동가이드.md).
        과도기에는 UDP(event_output.mode)와 병행 송신된다 — 웹소켓 정착 후 UDP 정리.
        """
        await websocket.accept()
        queue = asyncio.Queue()
        state.add_event_listener(asyncio.get_running_loop(), queue)
        try:
            while True:
                await websocket.send_text(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass   # 클라이언트 종료·연결 끊김 — 구독 해제만 하면 된다
        finally:
            state.remove_event_listener(queue)



    return app
