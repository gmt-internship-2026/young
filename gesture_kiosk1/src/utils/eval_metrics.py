"""정확도 평가 산식(2026-07-29) — 측정 세션(scripts/eval_accuracy.py)의 채점·집계·리포트.

회사 KPI №5(정확도 85%)의 산식을 코드로 고정한다:
    정확도 = 정답 시행 수 / 전체 시행 수   (오인식·미인식 모두 오답)
지시-수행 프로토콜(7종 이벤트 × N회, 무작위 순서)의 결과를 받아 전체 정확도,
제스처별 인식률, 혼동(의도→실제 출력) 목록을 마크다운으로 정리한다 — 튜닝 전후
비교·cam_a/cam_b 비교·회사 보고가 같은 숫자를 쓰게 하기 위한 단일 산식이다.
카메라·모델 없이 이 판정·집계만 단위 테스트한다 (tests/test_eval_metrics.py).
"""

RESULT_CORRECT = "correct"   # 지시한 이벤트가 제한 시간 안에 나옴
RESULT_WRONG = "wrong"       # 다른 이벤트가 나옴 (오인식 — 무엇이 나왔는지 기록)
RESULT_MISS = "miss"         # 제한 시간 안에 아무 이벤트도 없음 (미인식)


def judge_trial(expected_event, observed_event):
    """시행 1건 채점 — 관측된 **첫** 이벤트로 판정한다 (None = 미인식).

    첫 이벤트 기준인 이유: 키오스크에서는 첫 출력이 곧 화면 반응이다 —
    나중에 맞는 이벤트가 따라와도 사용자는 이미 잘못된 화면을 보고 있다.
    """
    if observed_event is None:
        return RESULT_MISS
    return RESULT_CORRECT if observed_event == expected_event else RESULT_WRONG


def aggregate_trials(trials):
    """시행 목록 [(지시 이벤트, 관측 이벤트|None)] -> 집계 dict.

    반환: total_count / correct_count / accuracy_ratio(KPI 산식) /
    per_event(이벤트별 시행·정답·오인식·미인식 수) /
    confusions(의도→오출력 쌍, 잦은 순) — 리포트와 검증이 같은 숫자를 쓴다.
    """
    per_event = {}
    confusion_counts = {}
    correct_count = 0
    for expected_event, observed_event in trials:
        stats = per_event.setdefault(expected_event, {
            "attempt_count": 0, "correct_count": 0, "wrong_count": 0, "miss_count": 0,
        })
        stats["attempt_count"] += 1
        result = judge_trial(expected_event, observed_event)
        if result == RESULT_CORRECT:
            stats["correct_count"] += 1
            correct_count += 1
        elif result == RESULT_WRONG:
            stats["wrong_count"] += 1
            pair = (expected_event, observed_event)
            confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
        else:
            stats["miss_count"] += 1
    total_count = len(trials)
    return {
        "total_count": total_count,
        "correct_count": correct_count,
        "accuracy_ratio": correct_count / total_count if total_count else 0.0,
        "per_event": per_event,
        "confusions": sorted(confusion_counts.items(), key=lambda item: -item[1]),
    }


def render_report(summary, meta):
    """집계(aggregate_trials) + 세션 정보(meta) -> 마크다운 리포트 문자열.

    meta: date / branch / reps_count / timeout_sec / stray_event_count / notes —
    재현 조건을 리포트에 같이 남겨 서로 다른 측정을 비교할 수 있게 한다.
    """
    lines = [
        f"# 제스처 정확도 측정 리포트 — {meta.get('date', '?')}",
        "",
        f"- 브랜치: {meta.get('branch', '?')}",
        f"- 프로토콜: 7종 × {meta.get('reps_count', '?')}회 무작위 순서 · "
        f"제한 시간 {meta.get('timeout_sec', '?')}초",
        f"- **전체 정확도: {summary['accuracy_ratio'] * 100:.1f}%** "
        f"({summary['correct_count']}/{summary['total_count']}) — KPI №5 산식",
        f"- 유휴 오발(지시 대기 중 발화): {meta.get('stray_event_count', 0)}건",
    ]
    if meta.get("notes"):
        lines.append(f"- 비고: {meta['notes']}")
    lines += [
        "",
        "## 제스처별 인식률",
        "",
        "| 이벤트 | 시행 | 정답 | 오인식 | 미인식 | 인식률 |",
        "|---|---|---|---|---|---|",
    ]
    for event_name, stats in summary["per_event"].items():
        attempt_count = stats["attempt_count"]
        ratio = stats["correct_count"] / attempt_count if attempt_count else 0.0
        lines.append(
            f"| {event_name} | {attempt_count} | {stats['correct_count']} "
            f"| {stats['wrong_count']} | {stats['miss_count']} | {ratio * 100:.0f}% |"
        )
    if summary["confusions"]:
        lines += ["", "## 혼동 (의도 → 실제 출력)", ""]
        for (expected_event, observed_event), pair_count in summary["confusions"]:
            lines.append(f"- {expected_event} → {observed_event}: {pair_count}회")
    lines.append("")
    return "\n".join(lines)
