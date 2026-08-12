"""postprocess 모듈 — 손 자세(모양+방향 콤보) 하나로 동작을 확정한다.

2026-08-05 신설(feat/shape_ml, 사용자 결정 — "완전 새로운 방식"): 종전
gesture_filter.GestureFilter는 손 모양(래치)과 이동 궤적(_SwipeTracker)을
따로 판정해 조합했다. 이 판은 "모양+방향" 콤보 라벨 하나(예: open_left)를
classify_pose_combo()가 매 프레임 뽑아주고, PoseGestureFilter는 궤적 추적
없이 **자세를 얼마나 안정적으로 유지했는가**만으로 이벤트를 확정한다 —
사용자 확인: "지금은 움직임이 필요없는거 같은데" (방향은 손목을 그 방향으로
기울인 정지 자세 자체가 표현하므로 이동 거리를 잴 필요가 없다).
hand_select.HandSelector·gesture_filter.GestureFilter는 건드리지 않는다 —
이 판만의 완전히 독립된 새 경로(기존 3모양+기하 방향 체계와 나란히 존재,
실시간 루프 연결은 아직 — 아래 classify_pose_combo 독스트링 참고).

상태기 구조(gesture_filter.shape_latch와 같은 이유— 프레임별 판별 출렁임이
그대로 판정을 오염시키면 안 된다):
① 후보 — 새 콤보가 연속 latch_frames 프레임 관측되면 "확정"(_confirmed_label,
   프레임 수 기반 — 아주 짧은(0.13초) 노이즈 제거 전담. 실제 발화 게이트는 ②)
② 발화 — 확정된 콤보가 이벤트로 정의돼 있고(COMBO_TO_EVENT) **그 콤보로
   확정된 채로 cooldown_sec(현재 1초)가 유지**되면 발화한다 — 콤보가 바뀌면
   (직전과 다른 자세로 전환하든, 같은 자세를 계속 들고 반복 발화를 노리든)
   매번 이 hold 시간을 처음부터 다시 채워야 한다.
   ★2026-08-07 3차 개정(사용자 결정 — "인식이 너무 빨라서 하고 싶지 않은
   제스처도 바로바로 잡혀 혼동이 온다 — 전체 다 1초로, 인식되고 1초 지나면
   실행, 연속 동작도 1초 틈이 생기게"): 직전엔(2026-08-07 2차 개정) "다른
   콤보 전환은 즉시, 같은 콤보 반복만 cooldown"으로 갈랐는데 — latch_frames
   (0.13초)만으로 확정되자마자 즉시 발화하다 보니, 다른 자세로 넘어가는
   **과정에 잠깐 스치는 중간 자세**까지 legit한 제스처로 발화돼 버렸다.
   이번 개정으로 "전환이든 반복이든 구분 없이 무조건 cooldown_sec만큼
   유지돼야 발화" — latch_frames는 여전히 1차 노이즈 필터(그 자세가 뭔지
   판별을 안정시키는 역할)로 남고, cooldown_sec이 "진짜 의도한 자세인지"를
   가르는 2차·최종 게이트가 됐다(2026-08-05 최초 도입 때의 "같은 콤보만
   cooldown" 스펙과도 다름 — 이번엔 콤보 동일 여부를 아예 안 본다).
   (2026-08-05 최초 도입 — "같은 방향 제스처를 계속 취하고 있으면 딜레이
   두고 한번 두번 세번 계속 인식": 반복 발화 자체의 취지는 유지 — 메뉴
   스크롤처럼 오래 들고 있을수록 여러 번 넘어가게. 발화 후 hold 타이머를
   다시 연 채로 남겨두면(아래 filter_signals) 계속 들고 있는 자세는
   cooldown_sec마다 알아서 반복 발화한다)
③ 해제 — 손 소실(release_sec 초과) 시에만 발화 이력(_fired_label, 계기판 표시용)을
   비운다(다음 손·다른 사용자에 승계 금지, hand_select 해제 규약과 동일 원칙).
   반복 발화 자체는 release_sec과 무관 — cooldown_sec만 본다
★2026-08-06 2차 개편(사용자 결정 — "내말대로 이제 다 바꾸어줘") — 아래가
**현재 유효한 스펙**이다(위 문단들의 top/bottom/home 배치는 이 개편으로
대체됨 — 역사 기록으로 남겨두되 지금 값은 COMBO_TO_EVENT 그대로 볼 것):

| 모양 | 방향/자세 | 이벤트 |
|---|---|---|
| open(손바닥) | left | left |
| open | right | right |
| open | up | **home** |
| open | down | **back** |
| fist(주먹, 방향 없음) | 정면 | **temp_fist**(3차 개편 — 아래 참고, 구 confirm) |
| fist | 손바닥 위(회전) | **temp_fist_back**(4차 개편 — 아래 참고, 구 fist_back) |
| finger(한 손가락) | up | **select**(3차 개편 — 구 click) |
| finger | right | temp_right |
| finger | down | **temp_bottom**(신설) |
| finger | left | temp_left |
| ok(엄지+검지 원, 나머지 폄, 방향 없음) | — | **confirm**(3차 개편 — 구 ok) |

fist_forward(주먹 앞으로)는 도입 하루 만에 폐기(2026-08-06 사용자 결정 —
"주먹쥐고 앞으로는 크게 의미없잖아 주먹모양이 중요한거니까"): 주먹 쥔 채
팔만 뻗는 변형은 fist와 구분할 실익이 없다고 판단, COMBO_TO_EVENT·
collect_gesture_pose_data.py의 forward 방향 키 모두 제거. fist_left/up/
right/down(방향 기울임 버전)도 여전히 스펙 밖 — 직전 개편에서 이미 뺐다.
finger_down은 이제 정의됨(temp_bottom) — down이 "정의 없음"인 조합은
fist_down 하나뿐이다(fist는 방향 대신 자세 2종만 씀).

ok(2026-08-06 사용자 요청 — "ok 제스처를 해서 ok라고 프로토콜"): 4번째
모양. fist처럼 방향 없는 단일 자세(collect_gesture_pose_data.py 모양 키
두 번 누르기)로 수집·판정한다.

★2026-08-06 3차 개편(사용자 결정 — "ok사인이 confirm 손가락 하나를
위로하면 select 정면 주먹쥐는 제스처를 temp_fist로 바꿔줘"): 라벨(콤보)은
그대로 두고 이벤트명 3개만 재배정 — ok→confirm(구 ok는 폐기, "확인"
의미를 ok 사인이 대신함), finger_up→select(구 click, 포커스 이동
의미로 재해석), fist→temp_fist(구 confirm, "확인"을 ok가 가져가서
plain fist는 임시 이름으로 격하 — 용도 미정). fist_back·나머지 조합은
불변.

★2026-08-11 4차 개편(사용자 결정): fist_back의 이벤트명만 fist_back→
temp_fist_back으로 재배정(plain fist가 temp_fist로 임시 이름 격하된 것과
같은 맥락 — "용도 미정" 상태를 이벤트명에도 반영). 라벨(콤보) 자체는
fist_back 그대로(collect_gesture_pose_data.py 수집 키·학습 데이터·분류기
클래스명 불변 — 이벤트명 재배정일 뿐 재학습 불필요). config.yaml의
classes: 목록도 함께 갱신.
"""
import time
from dataclasses import dataclass

from src.postprocess.hand_shape import SHAPE_FINGER, SHAPE_FIST, SHAPE_OPEN, classify_hand_shape
from src.postprocess.hand_shape_features import mirror_left_right_label
from src.utils.logger import get_logger

logger = get_logger("postprocess")

# 콤보 라벨("{모양}_{방향}", 방향 없는 순수 모양도 가능) -> 이벤트명.
# ★2026-08-11 4차 개편 스펙 — 모듈 독스트링의 표 참고. fist_left/up/right/down·
# fist_forward는 의도적으로 없음(방향 대신 자세로 temp_fist/temp_fist_back을 낸다)
COMBO_TO_EVENT = {
    "open_left": "left", "open_right": "right", "open_up": "home", "open_down": "back",
    "finger_left": "temp_left", "finger_right": "temp_right",
    "finger_up": "select", "finger_down": "temp_bottom",
    "fist": "temp_fist", "fist_back": "temp_fist_back",
    "ok": "confirm",
}

# 콤보 라벨의 모양 접두("{모양}_{방향}"에서 "{모양}") -> hand_shape.classify_hand_shape가
# 돌려주는 기하 판정 상수. ok는 기하 3분류(주먹/한손가락/손바닥) 밖의 자세라 대응이
# 없다 — geometric_shape_check에서 자동으로 검증 생략(아래 classify_pose_combo 참고)
_SHAPE_PREFIX_TO_GEOMETRIC = {"finger": SHAPE_FINGER, "fist": SHAPE_FIST, "open": SHAPE_OPEN}


def classify_pose_combo(classifier, hand, min_conf=None, max_dist_ratio=None,
                        none_margin=None, none_neighbor_ratio=None,
                        geometric_shape_check=None):
    """HandDetection 1건 -> 콤보 라벨("{모양}_{방향}") | None(min_conf·
    max_dist_ratio·none_margin·none_neighbor_ratio·geometric_shape_check 중
    하나라도 걸림 — 정의된 제스처 어디에도 안 닮음. HandShapeClassifier.classify
    독스트링 참고, 2026-08-07 사용자 요청 — "정의된 제스처 이외의 동작은
    철저하게 none으로 잡혀야").

    hand.user_side가 "left"면 랜드마크를 오른손 기준으로 미러링해 분류기에
    넣고(scripts/collect_gesture_pose_data.py가 오른손 데이터만으로 수집한
    전제와 짝을 맞춘다), 분류기 예측(오른손 기준 좌표계의 라벨)을 다시 한 번
    미러링해 **실제 방향**으로 되돌린다 — 왕복 미러링(수집 시 1회 + 추론 시
    1회)이 정확히 원래 동작을 복원한다(tests/test_hand_shape_features.py
    MirrorLeftHandTest로 좌표 반전 자체는 검증됨. 실제 왼손 데이터로 최종
    확인은 아직 — pose_gesture_filter.py 모듈 독스트링 참고).

    geometric_shape_check(2026-08-07 신설 — 사용자 요청 "손가락 갯수나 그냥
    진짜 아무것도 아닌 제스처... 학습 없이 되는 방법 없을까"): (extend_ratio,
    min_valid_fingers, curl_confirm_ratio) 튜플을 주면, 학습 분류기가 예측한
    모양이 실제 손가락 개수와 맞는지 hand_shape.classify_hand_shape()로
    교차검증한다 — 예를 들어 V사인(검지+중지 폄)은 이 함수가 extended_count=2라
    SHAPE_FINGER(정확히 1개 폄)도 SHAPE_OPEN(min_valid_fingers개 이상 폄)도
    아닌 None(불명)을 돌려주므로, 학습 분류기가 "finger_up"이라고 확신해도
    거부한다. **데이터 수집·재학습이 전혀 필요 없다** — min_conf·
    max_dist_ratio·none_margin(모두 학습 데이터 기반 통계적 방어)과 달리
    이건 순수 기하 규칙이라 한 번도 못 본 자세도 손가락 개수만 맞으면
    즉시 걸러진다. ok는 기하 3분류 밖이라(엄지+검지 원) 대응이 없어 검증을
    생략한다(_SHAPE_PREFIX_TO_GEOMETRIC 참고).
    ⚠ 반대급부: 기하 판정은 일부러 보수적이라(굽힘 미확인 손가락이 하나라도
    있으면 기권 — hand_shape.py 모듈 독스트링) 블러 등으로 가끔 진짜 제스처도
    한두 프레임 걸러질 수 있다 — PoseGestureFilter의 latch_frames+cooldown_sec
    다단 확인이 그 지터를 흡수하지만, 체감상 반응이 굼떠지면 이 옵션을 꺼서
    (config에서 키 삭제) 진단할 것.
    """
    is_left_hand = hand.user_side == "left"
    predicted = classifier.classify(hand.world_landmarks, is_left_hand=is_left_hand,
                                    min_conf=min_conf, max_dist_ratio=max_dist_ratio,
                                    none_margin=none_margin,
                                    none_neighbor_ratio=none_neighbor_ratio)
    if predicted is None:
        return None   # 정의 밖 동작 — 미러링할 라벨 자체가 없음
    label = mirror_left_right_label(predicted) if is_left_hand else predicted
    if geometric_shape_check is not None:
        expected_shape = _SHAPE_PREFIX_TO_GEOMETRIC.get(label.split("_", 1)[0])
        if expected_shape is not None:
            extend_ratio, min_valid_fingers, curl_confirm_ratio = geometric_shape_check
            geometric = classify_hand_shape(hand.world_landmarks, extend_ratio,
                                           min_valid_fingers, curl_confirm_ratio)
            if geometric != expected_shape:
                return None   # 학습 분류기와 손가락 개수가 안 맞음 — 정의 밖 동작
    return label


@dataclass
class PoseGestureEvent:
    """확정된 동작 이벤트 1건 — gesture_filter.GestureEvent와 같은 필드 구성
    (event_sender·realtime_loop가 그대로 재사용할 수 있게 맞춤)."""

    class_name: str
    conf: float
    ts_sec: float
    hand_side: str = None
    data: dict = None


class PoseGestureFilter:
    """모양+방향 콤보 라벨 스트림 -> 확정 이벤트. 궤적 추적 없음(모듈 독스트링)."""

    def __init__(self, config, clock=time.monotonic):
        pose_cfg = config["gestures"]["pose_classifier"]
        self._latch_frames = pose_cfg["latch_frames"]
        self._release_sec = pose_cfg["release_sec"]
        self._cooldown_sec = pose_cfg["cooldown_sec"]
        self._clock = clock

        self._candidate_label = None   # 연속 관측 세는 중인 콤보
        self._candidate_count = 0
        self._confirmed_label = None   # 래치된(안정적으로 유지 중인) 콤보
        self._confirmed_since_sec = None   # confirmed_label이 지금 값이 된 시각 —
                                        #   cooldown_sec(hold) 게이트의 기준(아래 filter_signals)
        self._fired_label = None       # 마지막으로 이벤트를 낸 콤보 — 계기판 표시용
        self._last_seen_sec = None     # 손이 마지막으로 관측된 시각 — release_sec 판단
        self.debug = {}

    def filter_signals(self, combo_label, hand_side=None):
        """combo_label: "{모양}_{방향}"(이미 왼손 미러링까지 끝난 실제 방향
        기준) | None(손 미관측 또는 판별 불가). hand_side: 이벤트 정보용 필드
        (참고용 — 정체성·판정에는 안 쓴다, gesture_filter와 같은 원칙).

        확정된 콤보(latch_frames로 노이즈를 거른 결과)가 이벤트로 정의돼
        있고(COMBO_TO_EVENT), **그 확정 상태가 cooldown_sec 이상 유지**됐으면
        이벤트 1건을 돌려준다 — 모듈 독스트링 ② 참고. 발화 즉시
        confirmed_since_sec을 지금 시각으로 다시 연다: 자세를 계속 들고
        있으면 confirmed_label이 안 바뀌니 latch_frames 재확정 없이 바로
        다음 cooldown_sec 카운트가 시작돼, 결과적으로 cooldown_sec 간격의
        반복 발화가 된다(모듈 독스트링 ② 후반 참고). confirmed_label이
        바뀌면(다른 자세로 전환) _update_candidate가 confirmed_since_sec을
        그 시점으로 리셋하므로, 전환도 동일하게 cooldown_sec을 처음부터
        기다려야 한다 — "전환은 즉시" 예외는 없다(2026-08-07 3차 개정).
        """
        now_sec = self._clock()
        if combo_label is None:
            self._update_absence(now_sec)
            self._update_debug()
            return None

        self._last_seen_sec = now_sec
        self._update_candidate(combo_label, now_sec)
        self._update_debug()

        if self._confirmed_label is None or self._confirmed_since_sec is None:
            return None
        if now_sec - self._confirmed_since_sec < self._cooldown_sec:
            return None   # 아직 hold 시간 못 채움 — 대기
        event_name = COMBO_TO_EVENT.get(self._confirmed_label)
        if event_name is None:
            return None   # 정의 없는 조합(주로 아래 방향) — 무시. confirmed_since_sec을
                          #   안 건드려야 다음에 진짜 콤보로 바뀌었을 때 hold를 다시
                          #   기다리지 않는다(정의 없는 조합은 "발화"가 아니다)
        self._fired_label = self._confirmed_label
        self._confirmed_since_sec = now_sec   # hold 타이머 재장전 — 계속 들고 있으면
                                              #   다음 cooldown_sec 뒤 반복 발화
        event = PoseGestureEvent(class_name=event_name, conf=1.0, ts_sec=now_sec,
                                 hand_side=hand_side)
        logger.info("pose_gesture_event: %s (combo=%s)", event_name, self._confirmed_label)
        return event

    def _update_candidate(self, combo_label, now_sec):
        """관측 1건 반영 — 연속 관측이 latch_frames를 넘으면 확정한다.

        확정 콤보가 실제로 **바뀌는** 순간에만 confirmed_since_sec을 지금
        시각으로 갱신한다(매 프레임 재확정될 때마다가 아니라) — 그래야
        filter_signals의 hold 게이트가 "이 콤보로 확정된 지 얼마나
        지났는가"를 정확히 잰다.
        """
        if combo_label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = combo_label
            self._candidate_count = 1
        if self._candidate_count >= self._latch_frames:
            if self._confirmed_label != combo_label:
                self._confirmed_since_sec = now_sec
            self._confirmed_label = combo_label

    def _update_absence(self, now_sec):
        """손 소실 — 후보 연속성은 매 소실 프레임마다 끊는다(다음 확정은 다시 latch_frames
        연속 관측부터). confirmed·fired는 release_sec을 넘겨야 비운다 — 짧은 블러 한두
        프레임에 이미 확정된 자세·발화 이력이 날아가 재발화(중복 이벤트)되면 안 된다.
        release_sec 초과는 다음 사용자에게 이전 발화 이력을 승계하지 않기 위함이다.
        """
        self._candidate_label = None
        self._candidate_count = 0
        if (self._last_seen_sec is not None
                and now_sec - self._last_seen_sec > self._release_sec):
            self._confirmed_label = None
            self._confirmed_since_sec = None
            self._fired_label = None

    def _update_debug(self):
        self.debug = {
            "candidate": (f"{self._candidate_label}:{self._candidate_count}"
                         if self._candidate_label else None),
            "confirmed": self._confirmed_label,
            "fired": self._fired_label,
        }
