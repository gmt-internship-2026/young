"""pose_gesture_filter 단위 테스트 — 카메라·모델 없이 상태기 로직만 검증한다
(feat/shape_ml: 모양+방향 12클래스 콤보 -> 이벤트, 궤적 추적 없음).
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.hand_shape_classifier import HandShapeClassifier
from src.postprocess.pose_gesture_filter import PoseGestureFilter, classify_pose_combo
from tests.hand_fixtures import make_hand

ALL_12_LABELS = [f"{shape}_{direction}"
                for shape in ("finger", "fist", "open")
                for direction in ("up", "down", "left", "right")]
FEATURE_COUNT = 60


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config(latch_frames=4, release_sec=1.5, cooldown_sec=0.6,
               repeat_events=("left", "right"), none_release_frames=None):
    # repeat_events 기본값은 configs/config.yaml의 실제 기본값(★2026-08-13
    # 5차 개편)과 맞춘다 — 대부분의 기존 테스트는 반복 여부를 안 다루므로
    # 이 기본값이 그대로 실 서비스 스펙과 일치해야 회귀 없이 의미가 통한다.
    # none_release_frames 기본 None(키 자체를 안 넣음) — PoseGestureFilter가
    # latch_frames로 대체하는 하위 호환 경로(★2026-08-20 7차 개편)를 그대로
    # 타서, 이 값을 안 다루는 기존 테스트들이 종전과 동일하게 동작한다
    cfg = {
        "latch_frames": latch_frames, "release_sec": release_sec, "cooldown_sec": cooldown_sec,
        "repeat_events": list(repeat_events),
    }
    if none_release_frames is not None:
        cfg["none_release_frames"] = none_release_frames
    return {"gestures": {"pose_classifier": cfg}}


class PoseGestureFilterTestBase(unittest.TestCase):
    """★2026-08-07 3차 개편(사용자 결정 — "인식이 너무 빨라서 하고 싶지 않은
    제스처도 바로바로 잡힌다 — 전체 다 1초로, 인식되고 1초 지나면 실행"):
    이제 latch_frames만으로는 발화하지 않는다 — 그 뒤로 cooldown_sec(hold)
    만큼 같은 콤보로 유지돼야 발화한다(pose_gesture_filter 모듈 독스트링 참고).
    `_feed`(frame_count만 공급)는 latch_frames 확인용으로만 남기고, 실제
    발화 확인은 `_feed_until_event`(콤보를 계속 공급하며 시계도 같이
    돌려 hold까지 채운다)를 쓴다.
    """

    def setUp(self):
        self.clock = FakeClock()
        self.filter = PoseGestureFilter(make_config(), clock=self.clock)

    def _feed(self, combo_label, frame_count=1, dt_sec=1.0 / 30.0):
        """frame_count 프레임 공급 — 그 안에 발화가 나오면 즉시 돌려준다(없으면 None).
        latch_frames 미만 공급처럼 "아직 확정도 안 됨"을 확인할 때 쓴다 —
        hold(cooldown_sec)까지 채우려면 `_feed_until_event`를 쓸 것."""
        for _ in range(frame_count):
            event = self.filter.filter_signals(combo_label)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None

    def _feed_until_event(self, combo_label, dt_sec=1.0 / 30.0, max_frames=200,
                          hand_side=None):
        """같은 콤보를 계속 공급 — latch_frames 확정 + cooldown_sec(hold)까지
        자연스럽게 채워 첫 발화를 얻는다. max_frames는 어떤 make_config 조합에도
        여유 있게 넉넉한 상한(200프레임 ≈ 6.7초)이라 진짜 못 채우면 None을 돌려준다."""
        for _ in range(max_frames):
            event = self.filter.filter_signals(combo_label, hand_side=hand_side)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None


class LatchConfirmTest(PoseGestureFilterTestBase):
    def test_needs_latch_frames_before_confirming(self):
        # latch_frames(4) 미만 — 아직 확정조차 안 됐으니 당연히 발화도 없다
        event = self._feed("open_left", frame_count=3)
        self.assertIsNone(event)

    def test_latch_frames_alone_is_not_enough_to_fire(self):
        # ★3차 개편 핵심 — 예전엔 latch_frames만 채우면 즉시 발화했지만,
        # 이제 그 뒤로 cooldown_sec(hold)까지 유지돼야 한다
        event = self._feed("open_left", frame_count=4)
        self.assertIsNone(event)

    def test_fires_once_confirmed_and_held_for_cooldown(self):
        event = self._feed_until_event("open_left")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_noisy_alternation_does_not_confirm(self):
        # 매 프레임 다른 라벨이면 연속 카운트가 계속 리셋돼 확정에 못 미친다
        for _ in range(10):
            event = self.filter.filter_signals("open_left")
            self.clock.tick(1.0 / 30.0)
            self.assertIsNone(event)
            event = self.filter.filter_signals("open_right")
            self.clock.tick(1.0 / 30.0)
            self.assertIsNone(event)


class RepeatWhileHeldTest(PoseGestureFilterTestBase):
    """2026-08-05 최초 도입 — "같은 방향 제스처를 계속 취하고 있으면 딜레이
    두고 한번 두번 세번 계속 인식": 자세를 계속 들고 있으면 cooldown_sec
    간격으로 반복 발화한다. ★2026-08-07 3차 개편으로 첫 발화 자체도
    cooldown_sec(hold)을 요구하게 됐지만, "계속 들고 있으면 반복 발화"라는
    이 취지는 그대로 유지된다 — 발화 즉시 hold 타이머가 재장전되기 때문
    (pose_gesture_filter.filter_signals 참고)."""

    def test_holding_same_pose_fires_then_refires_every_cooldown(self):
        first = self._feed_until_event("open_left")
        self.assertIsNotNone(first)
        self.assertEqual(first.class_name, "left")
        second = self._feed_until_event("open_left")   # 발화 직후부터 다시 hold
        self.assertIsNotNone(second)
        self.assertEqual(second.class_name, "left")

    def test_repeat_interval_matches_cooldown_sec_precisely(self):
        # ★2026-08-07 사용자 보고 — "간격이 1초보다 더 짧은 느낌인데?": 실제
        # 이벤트 타임스탬프(ts_sec) 차이를 cooldown_sec과 직접 비교해 정밀
        # 검증한다(기존 테스트는 "결국 발화하는지"만 봤음). dt_sec 오차 한
        # 프레임 이내여야 한다 — 그보다 짧으면 진짜 회귀 버그
        first = self._feed_until_event("open_left")
        second = self._feed_until_event("open_left")
        interval_sec = second.ts_sec - first.ts_sec
        self.assertGreaterEqual(interval_sec, self.filter._cooldown_sec)
        self.assertLess(interval_sec, self.filter._cooldown_sec + 1.0 / 30.0)

    def test_holding_same_pose_before_cooldown_elapsed_does_not_fire(self):
        self._feed("open_left", frame_count=4)   # 확정만 — hold는 아직
        for _ in range(10):   # 10프레임(1/30초 간격) ≈ 0.33초 — 쿨다운(0.6초) 안
            event = self.filter.filter_signals("open_left")
            self.clock.tick(1.0 / 30.0)
            self.assertIsNone(event)

    def test_changing_pose_requires_full_hold_before_firing(self):
        first = self._feed_until_event("open_left")
        self.assertEqual(first.class_name, "left")
        # 콤보가 바뀌면 hold 타이머가 리셋된다 — latch_frames만 채운 시점엔
        # 아직 발화하면 안 된다(★3차 개편 — 전환도 예외 없이 hold를 채워야 함)
        switch_event = self._feed("open_right", frame_count=4)
        self.assertIsNone(switch_event)
        # 이어서 hold를 마저 채우면 발화한다
        event = self._feed_until_event("open_right")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")


class FireOnceForNonRepeatEventsTest(PoseGestureFilterTestBase):
    """★2026-08-13 5차 개편(사용자 결정 — "left와 right만 1.5초 간격으로
    계속 받고, 나머지 제스처는 한 번 들어오면 다시 받지 않도록"):
    repeat_events(기본 left/right)에 없는 이벤트는 확정 후 딱 한 번만
    발화하고, 같은 콤보를 계속 들고 있어도 두 번째 발화가 없어야 한다."""

    def test_non_repeat_event_does_not_refire_while_held(self):
        first = self._feed_until_event("open_up")   # home — repeat_events 밖
        self.assertIsNotNone(first)
        self.assertEqual(first.class_name, "home")
        # cooldown_sec을 훌쩍 넘도록 같은 콤보를 계속 공급해도 재발화 없어야 한다
        for _ in range(60):   # 60프레임 * 1/30초 = 2초 > cooldown_sec(0.6초)
            event = self.filter.filter_signals("open_up")
            self.clock.tick(1.0 / 30.0)
            self.assertIsNone(event)

    def test_non_repeat_event_fires_again_after_switching_combo(self):
        first = self._feed_until_event("open_up")
        self.assertEqual(first.class_name, "home")
        # 다른 콤보로 전환됐다 되돌아오면(confirmed_label이 실제로 바뀜) 잠금 해제
        self._feed_until_event("open_down")
        second = self._feed_until_event("open_up")
        self.assertIsNotNone(second)
        self.assertEqual(second.class_name, "home")

    def test_non_repeat_event_fires_again_after_release(self):
        first = self._feed_until_event("finger_up")   # select — repeat_events 밖
        self.assertEqual(first.class_name, "select")
        self._feed(None, frame_count=60)   # release_sec(1.5초) 초과 소실
        second = self._feed_until_event("finger_up")
        self.assertIsNotNone(second)
        self.assertEqual(second.class_name, "select")

    def test_non_repeat_event_fires_again_after_brief_none_streak(self):
        # ★6차 개편(사용자 결정 — "다른 제스처뿐만 아니라 그냥 none 동작이
        # 되고 다시 제스처를 해도 되도록"): release_sec(1.5초)을 다 안
        # 기다려도, none이 none_release_frames만큼 연속 관측되면 잠금이
        # 풀린다(★7차 개편으로 이 문턱이 latch_frames와 분리됐지만, 기본값이
        # 키 미설정 시 latch_frames를 그대로 물려받으므로 이 테스트의 기본
        # config에서는 종전과 동일하게 동작한다)
        first = self._feed_until_event("open_up")
        self.assertEqual(first.class_name, "home")
        self._feed(None, frame_count=self.filter._none_release_frames)   # 짧은 none 스트릭
        second = self._feed_until_event("open_up")
        self.assertIsNotNone(second)
        self.assertEqual(second.class_name, "home")

    def test_single_none_frame_does_not_unlock(self):
        # none_release_frames 미만의 순간 none(오검출 등 노이즈 1프레임)만으로는
        # 잠금이 풀리면 안 된다 — 5차 개편의 "1회만 발화" 취지가 흔들림
        first = self._feed_until_event("open_up")
        self.assertEqual(first.class_name, "home")
        self._feed(None, frame_count=1)   # none_release_frames(기본 4)보다 훨씬 짧은 노이즈
        event = self._feed_until_event("open_up")
        self.assertIsNone(event)

    def test_repeat_events_configured_empty_makes_left_fire_once_too(self):
        self.filter = PoseGestureFilter(make_config(repeat_events=[]), clock=self.clock)
        first = self._feed_until_event("open_left")
        self.assertEqual(first.class_name, "left")
        for _ in range(60):
            event = self.filter.filter_signals("open_left")
            self.clock.tick(1.0 / 30.0)
            self.assertIsNone(event)

    def test_none_release_frames_defaults_to_latch_frames_when_unset(self):
        # ★7차 개편 하위 호환 — config에 none_release_frames 키가 없으면
        # 종전(latch_frames 재사용) 그대로 동작해야 한다
        self.filter = PoseGestureFilter(make_config(latch_frames=4), clock=self.clock)
        self.assertEqual(self.filter._none_release_frames, 4)

    def test_none_release_frames_separated_from_latch_frames(self):
        # ★7차 개편(2026-08-20, 사용자 보고 — "none이 너무 민감해서 조금만
        # 자세가 바뀌어도 none으로 인식하여 제스처를 여러 번 인식하는 문제"):
        # none_release_frames를 latch_frames보다 크게 두면, latch_frames 길이의
        # 순간적 none 스트릭(6차 개편 시절엔 이걸로 풀렸다)만으로는 더 이상
        # 잠금이 풀리지 않아야 한다 — 확정용/해제용 문턱이 독립됐기 때문
        self.filter = PoseGestureFilter(
            make_config(latch_frames=4, none_release_frames=10), clock=self.clock)
        first = self._feed_until_event("open_up")
        self.assertEqual(first.class_name, "home")
        self._feed(None, frame_count=4)   # 구 문턱(latch_frames)만큼만 none — 이제는 부족
        event = self._feed_until_event("open_up")
        self.assertIsNone(event)

    def test_none_release_frames_unlocks_once_its_own_threshold_reached(self):
        self.filter = PoseGestureFilter(
            make_config(latch_frames=4, none_release_frames=10), clock=self.clock)
        first = self._feed_until_event("open_up")
        self.assertEqual(first.class_name, "home")
        self._feed(None, frame_count=10)   # none_release_frames만큼 채움
        second = self._feed_until_event("open_up")
        self.assertIsNotNone(second)
        self.assertEqual(second.class_name, "home")

    def test_none_streak_unlock_still_requires_fresh_cooldown_hold(self):
        # ★8차 개편(2026-08-20, 사용자 실기 보고 — "아직 너무 민감해 한번에
        # 다라라락 올라와", 카메라 창 로그에서 select 연속 수십 건 확인):
        # 7차 개편만으로는 confirmed_label이 원래 라벨과 같을 때
        # confirmed_since_sec이 갱신 안 돼, none 스트릭에서 돌아오자마자
        # (latch_frames만 지나면) cooldown_sec 대기 없이 즉시 재발화됐다.
        # 이제 none 스트릭 완성 시점에 confirmed_since_sec도 재장전돼야 한다
        self.filter = PoseGestureFilter(
            make_config(latch_frames=4, cooldown_sec=0.6, none_release_frames=10),
            clock=self.clock)
        first = self._feed_until_event("open_up")
        self.assertEqual(first.class_name, "home")
        self._feed(None, frame_count=10)   # none_release_frames 채워 잠금 해제
        # 손이 돌아온 직후(latch_frames만 지난 시점)엔 아직 발화하면 안 된다 —
        # 8차 개편 전엔 여기서 즉시 재발화되던(폭주) 지점
        immediate = self._feed("open_up", frame_count=4)
        self.assertIsNone(immediate)
        # cooldown_sec까지 마저 채우면 그제서야 발화한다
        second = self._feed_until_event("open_up")
        self.assertIsNotNone(second)
        self.assertEqual(second.class_name, "home")

    def test_none_streak_unlock_does_not_cause_rapid_refire_burst(self):
        # 위 테스트의 실사용 시나리오 버전 — 자세가 확정 라벨과 none 사이를
        # 계속 오가도(카메라 앞에서 손이 살짝 흔들리는 상황 재현), cooldown_sec
        # 간격보다 짧게 여러 번 발화하면 안 된다(로그로 관측된 select 폭주 재현)
        self.filter = PoseGestureFilter(
            make_config(latch_frames=4, cooldown_sec=0.6, none_release_frames=10),
            clock=self.clock)
        fired_ts = []
        event = self._feed_until_event("open_up")
        fired_ts.append(event.ts_sec)
        for _ in range(6):   # none<->확정을 6번 오가며 재발화를 계속 시도
            self._feed(None, frame_count=10)
            event = self._feed("open_up", frame_count=20)
            if event is not None:
                fired_ts.append(event.ts_sec)
        for earlier, later in zip(fired_ts, fired_ts[1:]):
            self.assertGreaterEqual(later - earlier, self.filter._cooldown_sec)


class UndefinedComboTest(PoseGestureFilterTestBase):
    """2026-08-06 2차 개편 — open·finger는 4방향 다 정의됨, fist만 방향
    대신 자세(정면/손바닥 위) 2종이라 fist_down 등 방향 기울임 버전은
    여전히 정의 없음(모듈 독스트링 표 참고)."""

    def test_fist_direction_tilts_are_undefined(self):
        for direction in ("left", "up", "right", "down"):
            event = self._feed_until_event(f"fist_{direction}", max_frames=60)
            self.assertIsNone(event)

    def test_valid_combo_after_undefined_still_fires(self):
        self._feed("fist_down", frame_count=6)   # 정의 없음 — 확정 상태 안 건드림
        event = self._feed_until_event("fist_back")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "temp_fist_back")


class OpenPalmDirectionTest(PoseGestureFilterTestBase):
    """2026-08-06 2차 개편(사용자 결정 — "내말대로 이제 다 바꾸어줘"):
    손바닥 left/right는 그대로, up=home, down=back으로 재배치."""

    def test_open_up_fires_home(self):
        event = self._feed_until_event("open_up")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "home")

    def test_open_down_fires_back(self):
        event = self._feed_until_event("open_down")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "back")

    def test_open_left_and_right_unchanged(self):
        self.assertEqual(self._feed_until_event("open_left").class_name, "left")
        event = self._feed_until_event("open_right")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "right")


class FingerDirectionTest(PoseGestureFilterTestBase):
    """2026-08-06 2차 개편 — 한 손가락 4방향 전부 정의: up=click(신설),
    down=temp_bottom(신설), left/right는 그대로 temp_left/temp_right.
    ★3차 개편(사용자 결정 — "손가락 하나를 위로하면 select"): up의 이벤트명이
    click -> select로 재배정됐다(포커스 이동 의미로 재해석, 구 click은 소멸)."""

    def test_finger_up_fires_select(self):
        event = self._feed_until_event("finger_up")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "select")

    def test_finger_down_fires_temp_bottom(self):
        event = self._feed_until_event("finger_down")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "temp_bottom")

    def test_finger_left_and_right_unchanged(self):
        self.assertEqual(self._feed_until_event("finger_left").class_name, "temp_left")
        event = self._feed_until_event("finger_right")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "temp_right")


class FistPoseTest(PoseGestureFilterTestBase):
    """2026-08-06 2차 개편 — fist(정면, 방향 없는 단일 클래스), fist_back
    (손목 돌려 손바닥 위)=fist_back(당시 이벤트명이 라벨과 동일).
    fist_forward(주먹 앞으로)는 도입 하루 만에 폐기(사용자 결정 — "주먹쥐고
    앞으로는 크게 의미없잖아 주먹모양이 중요한거니까") — COMBO_TO_EVENT에서
    빠졌으므로 UndefinedComboTest처럼 정의 없는 조합으로 취급돼야 한다.
    ★3차 개편(사용자 결정 — "정면 주먹쥐는 제스처를 temp_fist로 바꿔줘"):
    plain fist의 이벤트명이 confirm -> temp_fist로 재배정됐다(구 confirm은
    ok 사인이 대신 가져감).
    ★2026-08-11 4차 개편(사용자 결정): fist_back의 이벤트명이 fist_back ->
    temp_fist_back으로 재배정됐다(라벨 자체는 fist_back 그대로 — 이벤트명만
    변경, pose_gesture_filter.py 모듈 독스트링 참고)."""

    def test_plain_fist_fires_temp_fist(self):
        event = self._feed_until_event("fist")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "temp_fist")

    def test_fist_forward_is_undefined(self):
        event = self._feed_until_event("fist_forward", max_frames=60)
        self.assertIsNone(event)

    def test_fist_back_fires_temp_fist_back_event(self):
        event = self._feed_until_event("fist_back")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "temp_fist_back")


class OkPoseTest(PoseGestureFilterTestBase):
    """2026-08-06 사용자 요청 — "ok 제스처를 해서 ok라고 프로토콜": fist처럼
    방향 없는 단일 자세로 수집·판정한다.
    ★3차 개편(사용자 결정 — "ok사인이 confirm"): 이벤트명이 도입 당일
    ok -> confirm으로 재배정됐다(구 confirm은 plain fist가 temp_fist로
    내주고 물려받음 — "확인" 의미는 이제 ok 사인이 전담)."""

    def test_ok_fires_confirm_event(self):
        event = self._feed_until_event("ok")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "confirm")


class ReleaseTest(PoseGestureFilterTestBase):
    def test_brief_dropout_does_not_reset_hold_timer(self):
        # latch만 채우고(아직 hold 중) 짧게 소실됐다 같은 콤보로 복귀 —
        # release_sec(1.5초) 안의 짧은 소실은 hold 진행 상태를 안 지운다
        # (confirmed_label이 안 바뀌므로 _update_candidate가 confirmed_since_sec을
        # 안 건드림) — 소실 없이 처음부터 채웠을 때와 같은 프레임 수 안에 발화한다
        self._feed("open_left", frame_count=4)
        self._feed(None, frame_count=5)   # release_sec(1.5초) 안의 짧은 소실
        event = self._feed_until_event("open_left")   # hold 이어서 채우면 발화
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_dropout_past_release_sec_requires_fresh_hold(self):
        self._feed_until_event("open_left")   # 1회 발화까지 완료
        self._feed(None, frame_count=60)   # 60프레임*1/30초 = 2초 > release_sec(1.5초)
        event = self._feed_until_event("open_left")   # 처음부터 다시 확정+hold
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "left")

    def test_dropout_resets_candidate_streak(self):
        # 확정 문턱(4) 직전에 1프레임 소실 — 후보 연속성이 끊겨 처음부터 다시 세야 한다
        self._feed("open_left", frame_count=3)
        self._feed(None, frame_count=1)
        event = self._feed("open_left", frame_count=3)   # 소실 전 3 + 이후 3, 하지만 연속 아님
        self.assertIsNone(event)


class CooldownTest(PoseGestureFilterTestBase):
    """★2026-08-07 3차 개편(사용자 결정 — "인식이 너무 빨라서 하고 싶지 않은
    제스처도 바로바로 잡혀 혼동이 온다 — 전체 다 1초로"): cooldown_sec은
    이제 전환·반복을 가리지 않는 단일 hold 게이트다 — 직전(2026-08-07 2차
    개편, "다른 콤보 전환은 즉시") 스펙은 이 개편으로 폐기됐다."""

    def test_switching_combo_still_requires_full_cooldown_hold(self):
        self._feed_until_event("open_left")
        # latch_frames만 채운 시점엔 아직 발화하면 안 된다 — 스치는 중간 자세가
        # 그대로 발화되던 문제(사용자 보고)의 재발 방지
        event = self._feed("open_right", frame_count=4)
        self.assertIsNone(event)

    def test_same_combo_repeat_within_cooldown_is_suppressed(self):
        self._feed_until_event("open_left")   # 1회 발화(hold 타이머 재장전됨)
        for _ in range(10):   # 10프레임(1/30초 간격) ≈ 0.33초 — 쿨다운(0.6초) 안
            event = self.filter.filter_signals("open_left")
            self.clock.tick(1.0 / 30.0)
            self.assertIsNone(event)


class EventFieldsTest(PoseGestureFilterTestBase):
    def test_hand_side_passed_through(self):
        event = self._feed_until_event("fist_back", hand_side="left")
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "temp_fist_back")
        self.assertEqual(event.hand_side, "left")


def _make_forced_classifier(forced_label):
    """어떤 입력을 넣어도 forced_label만 예측하는 합성 분류기 — coef=0, 그
    라벨의 intercept만 크게(classify_pose_combo의 좌우 미러링 왕복만 따로
    떼어 검증하려고, 실제 자세 판별 자체는 관여 안 하게 고정한다)."""
    tmpdir = tempfile.TemporaryDirectory()
    weights_path = os.path.join(tmpdir.name, "weights.npz")
    coef = np.zeros((len(ALL_12_LABELS), FEATURE_COUNT))
    intercept = np.zeros(len(ALL_12_LABELS))
    intercept[ALL_12_LABELS.index(forced_label)] = 100.0
    np.savez(weights_path, coef=coef, intercept=intercept, classes=np.array(ALL_12_LABELS))
    return HandShapeClassifier(weights_path), tmpdir


class ClassifyPoseComboMirrorTest(unittest.TestCase):
    """classify_pose_combo의 왼손 왕복 미러링(수집 1회 + 추론 1회 = 원복) —
    hand_shape_features.MirrorLeftHandTest는 좌표 반전 자체만 봤고, 여기서는
    HandShapeClassifier·hand_select.HandDetection과 엮인 통합 경로를 검증한다.
    """

    def test_right_hand_returns_predicted_label_unchanged(self):
        classifier, tmpdir = _make_forced_classifier("open_right")
        with tmpdir:
            hand = make_hand("right", "open")
            self.assertEqual(classify_pose_combo(classifier, hand), "open_right")

    def test_left_hand_predicted_label_is_mirrored_back(self):
        # 분류기는(오른손 기준 좌표계로 훈련됐다는 전제) 왼손이 들어와도 미러링된
        # 좌표를 보고 "open_right"를 예측한다(강제 고정) — classify_pose_combo는
        # 이걸 다시 미러링해 실제 방향인 "open_left"로 돌려줘야 한다
        classifier, tmpdir = _make_forced_classifier("open_right")
        with tmpdir:
            hand = make_hand("left", "open")
            self.assertEqual(classify_pose_combo(classifier, hand), "open_left")

    def test_up_down_labels_unaffected_by_hand_side(self):
        classifier, tmpdir = _make_forced_classifier("fist_up")
        with tmpdir:
            right_hand = make_hand("right", "fist")
            left_hand = make_hand("left", "fist")
            self.assertEqual(classify_pose_combo(classifier, right_hand), "fist_up")
            self.assertEqual(classify_pose_combo(classifier, left_hand), "fist_up")


def _make_forced_classifier_with_far_centroid(forced_label):
    """_make_forced_classifier와 같지만 centroids·max_dist도 저장 — 중심점을
    원점에, 반경을 아주 좁게(0.01) 둬서 실제 손 특징(원점과는 거리가 있음)이
    항상 반경 밖에 걸리게 만든다. "min_conf는 통과하지만(intercept로 강제
    고신뢰) 학습 데이터와는 안 닮은" 상황을 재현(max_dist_ratio 검증용)."""
    tmpdir = tempfile.TemporaryDirectory()
    weights_path = os.path.join(tmpdir.name, "weights.npz")
    coef = np.zeros((len(ALL_12_LABELS), FEATURE_COUNT))
    intercept = np.zeros(len(ALL_12_LABELS))
    intercept[ALL_12_LABELS.index(forced_label)] = 100.0
    centroids = np.zeros((len(ALL_12_LABELS), FEATURE_COUNT))
    max_dist = np.full(len(ALL_12_LABELS), 0.01)
    np.savez(weights_path, coef=coef, intercept=intercept, classes=np.array(ALL_12_LABELS),
            centroids=centroids, max_dist=max_dist)
    return HandShapeClassifier(weights_path), tmpdir


def _make_uniform_classifier():
    """모든 클래스가 동률(coef·intercept 전부 0)인 합성 분류기 — "정의된 어느
    제스처와도 안 닮은 입력"을 흉내낸다(2026-08-07 min_conf 신설분 검증용)."""
    tmpdir = tempfile.TemporaryDirectory()
    weights_path = os.path.join(tmpdir.name, "weights.npz")
    coef = np.zeros((len(ALL_12_LABELS), FEATURE_COUNT))
    intercept = np.zeros(len(ALL_12_LABELS))
    np.savez(weights_path, coef=coef, intercept=intercept, classes=np.array(ALL_12_LABELS))
    return HandShapeClassifier(weights_path), tmpdir


class ClassifyPoseComboMinConfTest(unittest.TestCase):
    """★2026-08-07 사용자 보고 — "정해진 제스처가 아닌데 비슷한 제스처로
    인식한다, 정의 밖 동작은 철저하게 none으로 잡혀야": min_conf를 넘겨주면
    확신 없는 예측을 None으로 거른다(HandShapeClassifier.classify 독스트링
    참고). classify_pose_combo가 None을 그대로 통과시키는지(미러링 시도로
    죽지 않는지)까지 확인한다."""

    def test_low_confidence_returns_none_right_hand(self):
        classifier, tmpdir = _make_uniform_classifier()
        with tmpdir:
            hand = make_hand("right", "open")
            self.assertIsNone(classify_pose_combo(classifier, hand, min_conf=0.6))

    def test_low_confidence_returns_none_left_hand(self):
        # 왼손 경로는 예측 -> 미러링 순서라, None이 미러링 함수로 들어가 죽지
        # 않는지가 핵심(mirror_left_right_label은 문자열 전제 — 가드 필요했음)
        classifier, tmpdir = _make_uniform_classifier()
        with tmpdir:
            hand = make_hand("left", "open")
            self.assertIsNone(classify_pose_combo(classifier, hand, min_conf=0.6))

    def test_high_confidence_unaffected_by_min_conf(self):
        classifier, tmpdir = _make_forced_classifier("open_right")
        with tmpdir:
            hand = make_hand("right", "open")
            self.assertEqual(classify_pose_combo(classifier, hand, min_conf=0.6), "open_right")

    def test_min_conf_none_default_never_rejects(self):
        classifier, tmpdir = _make_uniform_classifier()
        with tmpdir:
            hand = make_hand("right", "open")
            self.assertIsNotNone(classify_pose_combo(classifier, hand))   # min_conf 미지정


class ClassifyPoseComboMaxDistRatioTest(unittest.TestCase):
    """★2026-08-07 사용자 보고 — "여전히 다 잡아 이상한 제스처도": min_conf
    (신뢰도)만으론 선형 분류기의 무한 외삽을 못 막는다는 걸 여기서도
    재현 — intercept로 신뢰도를 강제로 100%에 가깝게 만들어도, 중심점까지
    거리(max_dist_ratio)는 별도로 걸러낸다."""

    def test_far_from_centroid_returns_none_despite_forced_high_confidence(self):
        classifier, tmpdir = _make_forced_classifier_with_far_centroid("open_right")
        with tmpdir:
            hand = make_hand("right", "open")
            self.assertEqual(classify_pose_combo(classifier, hand), "open_right")
            self.assertIsNone(classify_pose_combo(classifier, hand, max_dist_ratio=1.0))

    def test_max_dist_ratio_none_default_skips_distance_check(self):
        classifier, tmpdir = _make_forced_classifier_with_far_centroid("open_right")
        with tmpdir:
            hand = make_hand("right", "open")
            self.assertEqual(classify_pose_combo(classifier, hand), "open_right")


GEOMETRIC_CHECK_PARAMS = (1.35, 3, 0.9)   # (extend_ratio, min_valid_fingers,
                                          # curl_confirm_ratio) — hand_fixtures.py
                                          # 모듈 독스트링의 설계값과 동일


class ClassifyPoseComboGeometricCheckTest(unittest.TestCase):
    """★2026-08-07 사용자 요청 — "손가락 갯수나 그냥 진짜 아무것도 아닌
    제스처... 학습 없이 되는 방법 없을까": 학습 분류기가 뭐라고 예측하든,
    실제 편 손가락 개수가 그 모양과 안 맞으면 hand_shape.classify_hand_shape로
    거부한다 — V사인(검지+중지 폄)을 finger_up으로 강제 예측하게 만든
    분류기로도 실제로는 거부되는지 확인한다(데이터·재학습 불필요가 핵심)."""

    def test_v_sign_rejected_despite_forced_finger_prediction(self):
        classifier, tmpdir = _make_forced_classifier("finger_up")
        with tmpdir:
            hand = make_hand("right", "v_sign")   # 실제로는 손가락 2개 폄
            self.assertEqual(classify_pose_combo(classifier, hand), "finger_up")
            self.assertIsNone(classify_pose_combo(
                classifier, hand, geometric_shape_check=GEOMETRIC_CHECK_PARAMS))

    def test_matching_finger_count_passes_geometric_check(self):
        classifier, tmpdir = _make_forced_classifier("finger_up")
        with tmpdir:
            hand = make_hand("right", "finger")   # 실제로 검지 1개만 폄 — 일치
            result = classify_pose_combo(classifier, hand,
                                         geometric_shape_check=GEOMETRIC_CHECK_PARAMS)
            self.assertEqual(result, "finger_up")

    def test_geometric_check_none_default_skips_verification(self):
        classifier, tmpdir = _make_forced_classifier("finger_up")
        with tmpdir:
            hand = make_hand("right", "v_sign")
            self.assertEqual(classify_pose_combo(classifier, hand), "finger_up")

    def test_ok_combo_has_no_geometric_mapping_so_never_vetoed(self):
        # ok(엄지+검지 원)는 기하 3분류(주먹/한손가락/손바닥) 밖이라 손가락
        # 개수와 무관하게 검증을 생략해야 한다 — v_sign 픽스처로도 통과 확인.
        # ALL_12_LABELS엔 "ok"가 없어 _make_forced_classifier를 못 쓰므로
        # ["fist", "ok"] 2클래스 이진 분류기를 직접 만든다(HandShapeClassifier.
        # classify의 이진 분기 — classes[1]이 양성)
        tmpdir = tempfile.TemporaryDirectory()
        weights_path = os.path.join(tmpdir.name, "weights.npz")
        coef = np.zeros((1, FEATURE_COUNT))
        intercept = np.array([100.0])   # scores[0] > 0 -> classes[1]="ok" 강제
        np.savez(weights_path, coef=coef, intercept=intercept,
                classes=np.array(["fist", "ok"]))
        classifier = HandShapeClassifier(weights_path)
        with tmpdir:
            hand = make_hand("right", "v_sign")
            result = classify_pose_combo(classifier, hand,
                                         geometric_shape_check=GEOMETRIC_CHECK_PARAMS)
            self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
