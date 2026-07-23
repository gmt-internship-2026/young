"""postprocess 모듈 — 학습된 손 모양 분류기(로지스틱 회귀 가중치)로 point/fist/none을
판정한다. count_extended_fingers() 규칙 기반 방식의 대안 — 실기에서 규칙이 계속
불안정할 때만 config로 선택 켠다(gestures.shapes.classifier_weights_path 설정 시,
realtime_loop.py가 이 클래스를 만들어 count_extended_fingers() 대신 쓴다).

가중치는 numpy .npz 파일 하나뿐이라 학습 의존성(scikit-learn)이 추론 쪽에 필요
없다 — training/gesture/train_hand_shape.py가 학습·내보내기를 담당하고, 여기는
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

    def classify(self, landmarks):
        """21개 (x,y,z) 좌표 -> 클래스 이름 문자열(예: "point"/"fist"/"none")."""
        features = np.array(normalize_landmarks(landmarks), dtype=np.float64)
        scores = self._coef @ features + self._intercept
        if len(self._classes) == 2:
            # sklearn 이진 분류는 coef_가 (1, n_features) 하나뿐이라 다중 클래스
            # argmax와 규약이 다르다 — 양성 클래스(classes_[1]) 로짓 부호로 판정
            # (sklearn LogisticRegression 문서 기준)
            return self._classes[1] if scores[0] > 0 else self._classes[0]
        return self._classes[int(np.argmax(scores))]
