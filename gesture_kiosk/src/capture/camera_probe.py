"""capture 모듈 — 카메라 자동 선별(A안, 2026-07-28): 인식 품질로 메인 카메라를 고른다.

배포 키오스크에 웹캠이 2대 달리는 구성(№12 연계) 대응 — 장치 번호는 OS 열거
순서라 재부팅·포트 교체로 바뀔 수 있어 고정 device_id는 깨지기 쉽다. 시작할 때
각 장치를 잠깐 열어 **손 인식 품질**로 채점하면, 올바른 카메라가 번호와 무관하게
뽑히고 IR 카메라·가려진 카메라는 자동 탈락한다.

2026-07-29 포즈 제거: 채점도 손 품질 단독 — 얼굴(포즈) 감지율 항목 소멸.
점수 = 손 품질(크기×신뢰도) 평균 (2026-07-29 품질 채점 — 이진 감지는 앉은
사용자에서 위·아래 카메라가 동점이 되고, 동점은 낮은 번호가 이겨 구도 나쁜
카메라가 메인이 됐다. 손이 크게·또렷하게 보이는 카메라가 이긴다).

주의: 프로브 순간 카메라 앞에 사람이 손을 보여야 점수가 유효하다 — 아무도
없으면 전 장치 0점이라 config의 device_id를 그대로 쓴다(폴백).
score_probe_frames·_hand_quality는 순수 함수 — tests/test_camera_probe.py.
"""
import time

from src.capture.camera_stream import init_camera
from src.utils.logger import get_logger

logger = get_logger("capture")


def score_probe_frames(hand_frames):
    """프레임별 손 품질(0~1 — _hand_quality) 목록 -> 카메라 점수(0.0~1.0)."""
    if not hand_frames:
        return 0.0
    return sum(hand_frames) / len(hand_frames)


def _hand_quality(hands, frame_width_px, good_span_ratio):
    """프레임의 손 품질(0.0~1.0) — 가장 좋은 손의 (크기/기준)×신뢰도.

    크기 = 랜드마크 묶음의 최대 폭(px)/프레임 폭 — good_span_ratio에 도달하면
    크기 만점. 가깝고 정면으로(크게) 보이는 손이 높은 점수를 받아, 구도 좋은
    카메라가 이진 감지 동점을 깨고 이긴다 (2026-07-29 — 모듈 독스트링).
    """
    best_quality = 0.0
    for hand in hands:
        xs = hand.landmarks[:, 0]
        ys = hand.landmarks[:, 1]
        span_px = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        span_ratio = span_px / frame_width_px if frame_width_px else 0.0
        quality = min(1.0, span_ratio / good_span_ratio) * hand.conf
        best_quality = max(best_quality, quality)
    return best_quality


def rank_cameras(config, hand_tracker, preprocessor):
    """장치 0..N-1을 프로브 -> ([(device_id, 점수)] 내림차순, {device_id: 열린 cap}).

    cap을 닫지 않고 함께 돌려주는 이유(2026-07-28 실기): MSMF는 release 직후
    같은 장치를 다시 열면 프레임을 주지 않는다(오픈은 성공, read 무응답 —
    첫 프레임 타임아웃 크래시). 선택된 장치는 프로브가 연 핸들을 그대로
    재사용해야 한다 — 호출자는 select_camera를 쓰면 나머지가 정리된다.
    이미 만든 모델 인스턴스를 빌려 쓴다 — 프로브용 중복 로딩 방지.
    비활성(auto_select 없음/enabled false)이면 (빈 목록, 빈 dict).
    """
    probe_cfg = config["camera"].get("auto_select") or {}
    if not probe_cfg.get("enabled"):
        return [], {}
    ranked, caps = [], {}
    for device_id in range(probe_cfg["probe_device_count"]):
        result = _probe_device(config, device_id, probe_cfg, hand_tracker, preprocessor)
        if result is None:
            continue   # 장치 없음/열기 실패 — 후보 제외
        score, cap = result
        logger.info("카메라 프로브: device_id=%d 점수=%.2f", device_id, score)
        ranked.append((device_id, score))
        caps[device_id] = cap
    ranked.sort(key=lambda entry: entry[1], reverse=True)
    return ranked, caps


def select_camera(config, hand_tracker, preprocessor):
    """프로브 1위를 고른다 -> (device_id, 열린 cap | None).

    후보 없음·전원 0점이면 config의 device_id 폴백. 선택 장치의 cap은 열린
    채로 넘기고(MSMF 재오픈 무프레임 회피 — rank_cameras 주석) 나머지는 닫는다.
    """
    fallback = config["camera"]["device_id"]
    ranked, caps = rank_cameras(config, hand_tracker, preprocessor)
    if ranked and ranked[0][1] > 0.0:
        chosen = ranked[0][0]
        logger.info("카메라 자동 선별: device_id=%d (점수 %.2f)", chosen, ranked[0][1])
    else:
        chosen = fallback
        if config["camera"].get("auto_select", {}).get("enabled"):
            logger.warning("카메라 프로브 무효(후보 없음/전원 0점 — 손 미검출?) "
                           "— config device_id=%s 유지", fallback)
    chosen_cap = caps.pop(chosen, None)
    for cap in caps.values():
        cap.release()
    return chosen, chosen_cap


def _probe_device(config, device_id, probe_cfg, hand_tracker, preprocessor):
    """장치 1개를 열어 채점 -> (점수, 열린 cap) | None(열기 실패 — 장치 없음).

    시간 한도(probe_timeout_sec) 기반으로 읽는다 — MSMF는 오픈 직후 read 실패가
    흔해서(2026-07-28 실기: 시도 횟수 기반은 실패로만 소진돼 0점) 성공 프레임
    기준으로 워밍업·채점을 센다.
    """
    try:
        cap = init_camera(config, device_id=device_id)
    except RuntimeError:
        return None
    hand_frames = []
    warmup_left = probe_cfg["warmup_frames"]
    deadline_sec = time.monotonic() + probe_cfg.get("probe_timeout_sec", 3.0)
    while (len(hand_frames) < probe_cfg["probe_frames"]
           and time.monotonic() < deadline_sec):
        is_ok, frame = cap.read()
        if not is_ok:
            continue
        if warmup_left > 0:
            warmup_left -= 1   # 자동 노출 안정화 전 프레임은 채점 제외
            continue
        frame = preprocessor.preprocess_frame(frame)
        hands = hand_tracker.infer(frame)
        hand_frames.append(_hand_quality(
            hands, frame.shape[1], probe_cfg.get("good_hand_span_ratio", 0.10)))
    return score_probe_frames(hand_frames), cap
