"""정확도 측정 세션 — 지시대로 동작하면 자동 채점해 리포트를 남긴다 (2026-07-29).

KPI №5(정확도 85%) 산식 확보·튜닝 회귀 검증·cam_a/cam_b 비교의 공통 측정 도구.
지시 창(cv2)에 동작이 큰 글씨로 표시되고(6종 × N회, 무작위 순서 — 2026-07-29
top/bottom 제거·select/confirm 개편, 2026-08-04 한 손가락↔손바닥 명칭 맞교환
(구 select → temp_top) 반영), 제한 시간 안에
나온 **첫** 이벤트로 채점한다 — 정답 / 오인식(무엇이 나갔는지) / 미인식.
결과는 logs/eval_*.md 마크다운 리포트로 저장된다 (산식: src/utils/eval_metrics.py).

사용법 (프로젝트 루트에서 — 더블클릭은 eval.bat):
    python scripts/eval_accuracy.py [--reps 3] [--timeout-sec 6.0]
중단: 지시 창에서 q/ESC — 그때까지의 시행으로 부분 리포트를 남긴다.
"""
import argparse
import datetime
import os
import random
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

from src.utils.config_loader import load_config
from src.utils.eval_metrics import RESULT_CORRECT, RESULT_MISS, aggregate_trials, judge_trial, render_report
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
WINDOW_NAME = "gesture_kiosk eval"
READY_SEC = 2.0        # 시행 사이 준비 시간 — 손 내리고 재정렬 (쿨다운·복귀 삼킴 소화)
FEEDBACK_SEC = 1.0     # 채점 결과 표시 시간

# 지시 목록 — (이벤트, 지시 창 표기, 콘솔 한국어). 창 표기는 ASCII —
# cv2 기본 폰트(Hershey)는 한글을 못 그린다 (별도 폰트 의존을 더하지 않는다)
INSTRUCTIONS = [
    ("temp_left", "FINGER  <--", "한 손가락 · 왼쪽"),
    ("temp_right", "FINGER  -->", "한 손가락 · 오른쪽"),
    ("temp_top", "FINGER  UP  [temp_top]", "한 손가락 · 위 = 임시 계층"),
    ("back", "FIST  <--   [back]", "주먹 · 왼쪽 = 이전"),
    ("home", "FIST  UP    [home]", "주먹 · 위 = 처음으로"),
    ("confirm", "FIST  -->   [confirm]", "주먹 · 오른쪽 = 확인"),
]

logger = get_logger("scripts")


def read_git_branch():
    """현재 브랜치 이름 — 리포트에 측정 대상(판)을 남긴다.

    .git 없음(반출 폴더 — USB 반입 키오스크) 시 "unknown": '?'는 윈도우 파일명
    금지 문자라 리포트 저장이 터진다 (2026-07-29 반출 검증에서 발견·수정).
    """
    try:
        with open(os.path.join(ROOT_DIR, "..", ".git", "HEAD"), encoding="utf-8") as head_file:
            head = head_file.read().strip()
        return head.rsplit("/", 1)[-1] if head.startswith("ref:") else head[:8]
    except OSError:
        return "unknown"


def draw_panel(main_text, sub_text, color, progress_text=""):
    """지시 창 화면 1장 — 큰 지시문 + 보조 설명 + 진행도."""
    canvas = np.zeros((480, 900, 3), dtype=np.uint8)
    cv2.putText(canvas, main_text, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.9, color, 5)
    cv2.putText(canvas, sub_text, (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    if progress_text:
        cv2.putText(canvas, progress_text, (40, 440), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (140, 140, 140), 2)
    return canvas


def wait_phase(state, canvas, duration_sec, baseline_count, stop_on_event):
    """canvas를 띄운 채 대기 — (관측 이벤트|None, 새 이벤트 수, 중단 여부).

    stop_on_event=True(지시 구간)면 첫 이벤트에서 즉시 끝난다.
    False(준비 구간)면 끝까지 기다리며 유휴 오발만 센다. q/ESC = 세션 중단.
    """
    deadline_sec = time.monotonic() + duration_sec
    observed_event = None
    while time.monotonic() < deadline_sec:
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):   # 27 = ESC
            return observed_event, len(state.event_log) - baseline_count, True
        if stop_on_event and len(state.event_log) > baseline_count:
            observed_event = state.event_log[baseline_count].class_name
            break
    return observed_event, len(state.event_log) - baseline_count, False


def run_eval(state, reps_count, timeout_sec):
    """측정 본체 — 시행 목록 [(지시, 관측|None)]과 유휴 오발 수를 돌려준다."""
    trial_plan = INSTRUCTIONS * reps_count
    random.shuffle(trial_plan)   # 순서 학습(다음 동작 예측) 방지
    trials = []
    stray_event_count = 0
    for trial_idx, (event_name, big_text, korean_text) in enumerate(trial_plan):
        progress = f"{trial_idx + 1} / {len(trial_plan)}   (q/ESC = stop)"
        # 준비 구간 — 손 내리고 대기. 이 동안 나온 이벤트는 유휴 오발로 센다
        ready_canvas = draw_panel("READY - hand down", "손을 내리고 기다리세요",
                                  (160, 160, 160), progress)
        _, ready_events, aborted = wait_phase(
            state, ready_canvas, READY_SEC, len(state.event_log), stop_on_event=False)
        stray_event_count += ready_events
        if aborted:
            break
        # 지시 구간 — 첫 이벤트로 채점
        logger.info("측정 %d/%d — 지시: %s (%s)",
                    trial_idx + 1, len(trial_plan), event_name, korean_text)
        instruct_canvas = draw_panel(big_text, korean_text, (255, 255, 255), progress)
        observed_event, _, aborted = wait_phase(
            state, instruct_canvas, timeout_sec, len(state.event_log), stop_on_event=True)
        trials.append((event_name, observed_event))
        result = judge_trial(event_name, observed_event)
        if result == RESULT_CORRECT:
            feedback = draw_panel("OK", f"{event_name}", (80, 220, 80), progress)
        elif result == RESULT_MISS:
            feedback = draw_panel("MISS", "no event", (60, 160, 255), progress)
        else:
            feedback = draw_panel("WRONG", f"{event_name} -> {observed_event}",
                                  (60, 60, 255), progress)
        _, feedback_events, feedback_abort = wait_phase(
            state, feedback, FEEDBACK_SEC, len(state.event_log), stop_on_event=False)
        stray_event_count += feedback_events
        if aborted or feedback_abort:
            break
    return trials, stray_event_count


def main():
    parser = argparse.ArgumentParser(description="gesture_kiosk 정확도 측정 세션")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--reps", type=int, default=3, help="이벤트당 시행 수 (6종 × N)")
    parser.add_argument("--timeout-sec", type=float, default=6.0, help="시행당 제한 시간")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)

    from src.pipeline.realtime_loop import run_pipeline

    state = run_pipeline(config)
    logger.info("측정 세션 시작 — 지시 창을 보며 동작하세요 (6종 × %d회)", args.reps)
    try:
        trials, stray_event_count = run_eval(state, args.reps, args.timeout_sec)
    finally:
        state.is_running = False
        cv2.destroyAllWindows()

    if not trials:
        logger.warning("시행 0건 — 리포트를 만들지 않습니다")
        return 1
    summary = aggregate_trials(trials)
    now = datetime.datetime.now()
    meta = {
        "date": now.strftime("%Y-%m-%d %H:%M"),
        "branch": read_git_branch(),
        "reps_count": args.reps,
        "timeout_sec": args.timeout_sec,
        "stray_event_count": stray_event_count,
        "notes": "" if len(trials) == args.reps * len(INSTRUCTIONS) else
                 f"중단됨 — {len(trials)}건까지의 부분 측정",
    }
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(
        save_dir, f"eval_{now.strftime('%Y-%m-%d_%H%M')}_{meta['branch']}.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(render_report(summary, meta))
    logger.info("전체 정확도 %.1f%% (%d/%d) · 유휴 오발 %d건 — 리포트: %s",
                summary["accuracy_ratio"] * 100, summary["correct_count"],
                summary["total_count"], stray_event_count, report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
