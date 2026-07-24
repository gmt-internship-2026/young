"""postprocess 모듈 — 학습된 방향 분류기(로지스틱 회귀 가중치)로 손 위치 궤적의 방향
(left/right/up/down/none)을 판정한다. _SwipeTracker의 임계값 비교(min_dist_*_ratio·
axis_dominance) 방식의 대안 — 실기에서 임계값 재튜닝으로도 오판정이 계속될 때만 config로
선택 켠다(gestures.hand_move.classifier_weights_path 설정 시, realtime_loop.py가 이
클래스를 만들어 _SwipeTracker에 주입한다).

가중치는 numpy .npz 파일 하나뿐이라 학습 의존성(scikit-learn)이 추론 쪽에 필요 없다 —
scripts/train_direction_classifier.py가 학습·내보내기를 담당하고, 여기는 행렬곱 하나로
추론한다(hand_shape_classifier.py와 같은 원칙).
"""
import numpy as np

from src.postprocess.direction_features import extract_window_features


class DirectionClassifier:
    """학습된 로지스틱 회귀 가중치로 손 위치 궤적의 방향을 분류한다."""

    def __init__(self, weights_path):
        data = np.load(weights_path, allow_pickle=False)
        self._coef = data["coef"]            # (n_classes, n_features) 또는 이진이면 (1, n_features)
        self._intercept = data["intercept"]  # (n_classes,) 또는 (1,)
        self._classes = [str(c) for c in data["classes"]]

    def classify(self, track):
        """[(ts_sec, x_ratio, y_ratio), ...] -> 클래스 이름 문자열("left"/"right"/"up"/"down"/"none")."""
        features = np.array(extract_window_features(track), dtype=np.float64)
        scores = self._coef @ features + self._intercept
        if len(self._classes) == 2:
            # sklearn 이진 분류는 coef_가 (1, n_features) 하나뿐이라 다중 클래스
            # argmax와 규약이 다르다 — 양성 클래스(classes_[1]) 로짓 부호로 판정
            # (sklearn LogisticRegression 문서 기준)
            return self._classes[1] if scores[0] > 0 else self._classes[0]
        return self._classes[int(np.argmax(scores))]
