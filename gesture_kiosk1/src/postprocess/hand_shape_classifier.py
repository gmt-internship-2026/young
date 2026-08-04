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

    @property
    def classes(self):
        """학습된 클래스 이름 목록 — 특정 클래스(예: "open")를 아직 모르는
        구버전 가중치인지 호출 쪽(hand_select.py)이 확인하는 용도."""
        return self._classes

    def classify(self, landmarks):
        """21개 (x,y,z) 좌표 -> 클래스 이름 문자열(예: "finger"/"fist"/"open"/"none")."""
        features = np.array(normalize_landmarks(landmarks), dtype=np.float64)
        scores = self._coef @ features + self._intercept
        if len(self._classes) == 2:
            # sklearn 이진 분류는 coef_가 (1, n_features) 하나뿐이라 다중 클래스
            # argmax와 규약이 다르다 — 양성 클래스(classes_[1]) 로짓 부호로 판정
            # (sklearn LogisticRegression 문서 기준)
            return self._classes[1] if scores[0] > 0 else self._classes[0]
        return self._classes[int(np.argmax(scores))]
