"""postprocess 모듈 — 학습된 손 모양 분류기(로지스틱 회귀 가중치)로 손 모양을
판정한다. hand_shape.classify_hand_shape() 규칙 기반 방식의 보조/대체 — 실기에서
손가락 개수 판별이 흔들릴 때 config로 선택 켠다(hand_select.hand_shape.
classifier_weights_path 설정 시, hand_select.py가 이 클래스를 만들어 기하 판정과
같이 쓴다 — 통합 방식은 hand_select.py 주석 참고). 2026-08-03 CPU 브랜치에서 이식.

가중치는 numpy .npz 파일 하나뿐이라 학습 의존성(scikit-learn)이 추론 쪽에 필요
없다 — scripts/train_hand_shape_classifier.py가 학습·내보내기를 담당하고, 여기는
행렬곱 하나로 추론한다(무거운 의존 지연 임포트 원칙과 같은 이유 — 애초에 무거운
의존 자체가 없다).
"""
import numpy as np

from src.postprocess.hand_shape_features import normalize_landmarks


class HandShapeClassifier:
    """학습된 로지스틱 회귀 가중치로 손 모양을 분류한다."""

    def __init__(self, weights_path):
        data = np.load(weights_path, allow_pickle=False)
        self._coef = data["coef"]            # (n_classes, n_features) 또는 이진이면 (1, n_features)
        self._intercept = data["intercept"]  # (n_classes,) 또는 (1,)
        self._classes = [str(c) for c in data["classes"]]
        # 2026-08-07 신설(구버전 .npz 호환 — 없으면 max_dist_ratio 검사를 건너뜀)
        self._centroids = data["centroids"] if "centroids" in data.files else None
        self._max_dist = data["max_dist"] if "max_dist" in data.files else None
        # 2026-08-07 신설(구버전·none 미학습 .npz 호환 — 없으면 none_neighbor_ratio
        # 검사를 건너뜀). none_class_idx는 매 classify() 호출마다 다시 안 찾게 캐싱
        self._none_features = data["none_features"] if "none_features" in data.files else None
        self._none_typical_gap = (float(data["none_typical_gap"])
                                  if "none_typical_gap" in data.files else None)
        self._none_class_idx = self._classes.index("none") if "none" in self._classes else None

    @property
    def classes(self):
        """학습된 클래스 이름 목록 — 특정 클래스(예: "open")를 아직 모르는
        구버전 가중치인지 호출 쪽(hand_select.py)이 확인하는 용도."""
        return self._classes

    def classify(self, landmarks, is_left_hand=False, min_conf=None, max_dist_ratio=None,
                none_margin=None, none_neighbor_ratio=None):
        """21개 (x,y,z) 좌표 -> 클래스 이름 문자열(예: "finger"/"fist"/"open"/"none")
        | None(min_conf·max_dist_ratio·none_margin·none_neighbor_ratio 중
        하나라도 걸리면 — 아래 참고).

        is_left_hand(2026-08-05 feat/shape_ml): normalize_landmarks에 그대로
        전달 — 왼손이면 오른손 기준 좌표계로 미러링해 넣는다(pose_gesture_filter.
        classify_pose_combo 참고). 기본 False라 기존 3모양(open/finger/fist)
        판정(hand_select.py — 방향이 없어 미러링이 필요 없다)은 그대로 동작한다.

        min_conf(2026-08-07 사용자 보고 — "정해진 제스처가 아닌데 비슷한
        제스처로 인식한다, 정의 밖 동작은 철저하게 none으로 잡혀야"):
        로지스틱 회귀는 항상 학습된 클래스 중 하나를 강제로 골라야 하는
        구조라(argmax에 "모르겠다"가 없음) 정의 밖 자세도 그중 가장 가까운
        클래스로 우겨넣는다. min_conf를 주면 승자 클래스의 확률(소프트맥스/
        시그모이드)이 이 값 미만일 때 None을 돌려준다.
        ⚠ 실기에서 이것만으론 부족했다(사용자 보고 — "여전히 다 잡아 이상한
        제스처도"): 로지스틱 회귀는 선형 경계이므로 경계에서 멀어질수록
        로짓이 오히려 커진다 — 학습 데이터와 전혀 안 닮은 입력도 우연히
        어느 클래스 방향으로 멀리 있으면 min_conf를 가볍게 넘는 확신을
        낸다(선형 분류기의 무한 외삽). min_conf 단독으론 이 케이스를 못 거른다.

        max_dist_ratio(2026-08-07 신설 — min_conf의 한계를 보완): "예측
        자체"가 아니라 "학습 데이터와 실제로 얼마나 떨어져 있는가"를 본다.
        학습 시(scripts/train_hand_shape_classifier.py) 클래스별 중심점
        (centroids)과 반경(max_dist, 그 클래스 학습 샘플 중 중심점에서 가장
        먼 거리)을 같이 저장해두면, 추론 시 입력 특징에서 **가장 가까운
        중심점까지의 실제 거리**가 그 클래스 반경 × max_dist_ratio를 넘을 때
        None을 돌려준다 — 로지스틱 회귀의 무한 외삽과 무관하게 "본 적 있는
        데이터 근방인가"만 본다. 구버전 .npz(centroids 없음)에서는 이 검사를
        건너뛴다(하위 호환) — 재학습해야 실제로 적용된다.

        none_margin(2026-08-07 신설 — 사용자 요청 "추론값으로도 좀 해당
        제스처만 잡도록", "none" 클래스가 실제로 학습된 뒤부터 유효): 위
        두 검사는 "승자 클래스가 얼마나 확신하는가/가까운가"만 보고 "none이
        2등으로 얼마나 바짝 따라붙었는가"는 안 본다 — 승자가 min_conf를
        넉넉히 넘어도 none이 근소한 차이의 2등이면 사실상 애매한 경우다.
        none_margin을 주면, 승자가 "none"이 아닐 때 (승자 확률 - none 확률)이
        이 값 미만이면 None을 돌려준다 — none과 충분히 벌어져야만 그 제스처를
        인정. "none" 클래스가 없는 가중치(구버전 hand_select 3모양 등)에서는
        건너뛴다(하위 호환).

        none_neighbor_ratio(2026-08-07 신설 — max_dist_ratio가 "none" 클래스엔
        잘 안 맞는 문제 보완): none은 서로 완전히 다른 자세(V사인·중지만·
        손흔들기 등)를 한 라벨로 뭉친 다봉(multi-modal) 클래스라, 중심점(평균)이
        애매한 값이 되고 반경(max_dist)은 그 다양성을 다 덮으려 크게 부풀어
        max_dist_ratio가 사실상 무력화된다(실기 확인 — 뻐큐는 걸러졌는데 V사인은
        여전히 통과). 이건 "none 전체 평균과의 거리"가 아니라 **학습 시 저장해둔
        개별 none 샘플들 중 가장 가까운 것 하나**까지 거리를 본다 — 그 거리가
        none_typical_gap(학습 시 계산된, none 샘플들끼리도 서로 떨어져 있는
        전형적 거리) × 이 배수보다 가까우면 None을 돌려준다. "본 적 있는 none
        샘플 근방인가"만 보므로 뭉뚱그려진 none 클래스에도 안 흔들린다.
        재학습해서 none 데이터를 실제로 넣어야(scripts/
        train_hand_shape_classifier.py) 의미가 생기고, none 클래스가 아예
        없으면 건너뛴다(하위 호환).

        네 검사 모두 기본값 None(안 줌)이면 종전 그대로(항상 argmax 강제) —
        hand_select.py의 기존 호출은 인자를 안 넘기므로 전혀 영향 없다
        (pose_classifier 엔진 경로에만 적용, classify_pose_combo가 threading).
        """
        features = np.array(normalize_landmarks(landmarks, is_left_hand=is_left_hand),
                            dtype=np.float64)
        scores = self._coef @ features + self._intercept
        if len(self._classes) == 2:
            # sklearn 이진 분류는 coef_가 (1, n_features) 하나뿐이라 다중 클래스
            # argmax와 규약이 다르다 — 양성 클래스(classes_[1]) 로짓 부호로 판정
            # (sklearn LogisticRegression 문서 기준). 시그모이드로 두 클래스
            # 확률을 다 구해 아래 다중클래스 경로와 같은 형태(probs 배열)로 맞춘다
            positive_prob = 1.0 / (1.0 + np.exp(-scores[0]))
            probs = np.array([1.0 - positive_prob, positive_prob])
        else:
            exp_scores = np.exp(scores - scores.max())   # 오버플로 방지 — 최댓값 빼고 지수화
            probs = exp_scores / exp_scores.sum()
        idx = int(np.argmax(probs))
        predicted, conf = self._classes[idx], float(probs[idx])
        if min_conf is not None and conf < min_conf:
            return None
        if (none_margin is not None and predicted != "none"
                and self._none_class_idx is not None):
            none_prob = float(probs[self._none_class_idx])
            if conf - none_prob < none_margin:
                return None
        if max_dist_ratio is not None and self._centroids is not None:
            dists = np.linalg.norm(self._centroids - features, axis=1)
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] > self._max_dist[nearest_idx] * max_dist_ratio:
                return None
        if (none_neighbor_ratio is not None and self._none_features is not None
                and predicted != "none"):
            nearest_none_dist = float(
                np.linalg.norm(self._none_features - features, axis=1).min())
            if nearest_none_dist <= self._none_typical_gap * none_neighbor_ratio:
                return None
        return predicted
