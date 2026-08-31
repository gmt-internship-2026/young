"""inference 전처리 — 추론 입력용으로 프레임을 가공한다 (기획서 4.6 계약).

현재 백엔드(ultralytics)는 리사이즈·정규화를 내부에서 처리하므로
여기서는 거울 반전과 밝기 자동 보정만 담당한다. 추후 TensorRT 바인딩을
직접 다루게 되면 letterbox·정규화·CHW 변환이 이 모듈로 들어온다.

밝기 자동 보정(2026-08-13 저조도 전용 감마 보정으로 신설, 2026-08-21 양방향
상시 보정으로 확장 — 사용자 요청 "항상 일정한 밝기를 유지"): 어두운 곳에서
손이 배경에 묻히거나, 반대로 역광 등으로 과다 노출되면 둘 다 HandLandmarker
검출률이 떨어진다. 노출 시간을 늘리는 카메라 쪽 보정은 쓸기 모션 블러를
키워 역효과라(2026-08-13 결정 유지), 대신 매 프레임 평균 밝기를
target_luma로 조금씩(gamma_step) 끌어당기는 감마 보정을 상시 적용한다 —
deadband 안이면 손대지 않고(미세 흔들림 방지), 감마가 바뀔 때만 LUT를
다시 만든다. 임계값·감마 범위는 현장 실측 전 잠정값 — configs/config.yaml
brightness, docs/TODO.md 참고.
"""
import cv2
import numpy as np


class Preprocessor:
    def __init__(self, config):
        self._is_mirror = config["camera"]["mirror"]
        brightness_config = config.get("brightness", {})
        self._is_brightness_enabled = brightness_config.get("enabled", False)
        self._target_luma = brightness_config.get("target_luma", 0.45)
        self._luma_deadband = brightness_config.get("luma_deadband", 0.05)
        self._gamma_step = brightness_config.get("gamma_step", 0.05)
        self._gamma_min = brightness_config.get("gamma_min", 0.5)
        self._gamma_max = brightness_config.get("gamma_max", 2.2)
        # 목표를 향해 매 프레임 한 걸음씩만 이동하는 현재 감마 — 생성자에서는
        # 무보정(1.0)으로 시작해 첫 몇 프레임 동안 목표로 수렴한다
        self._current_gamma = 1.0
        self._gamma_lut = self._build_gamma_lut(self._current_gamma)

    def preprocess_frame(self, frame):
        """frame(BGR) -> input_tensor. 거울 모드면 좌우 반전, 밝기 보정이 켜져 있으면
        평균 밝기를 목표값 쪽으로 계속 맞춘다."""
        if self._is_mirror:
            frame = cv2.flip(frame, 1)
        if self._is_brightness_enabled:
            frame = self._apply_brightness_correction(frame)
        return frame

    def _apply_brightness_correction(self, frame):
        luma_error = self._target_luma - self._measure_mean_luma(frame)
        if abs(luma_error) > self._luma_deadband:
            step = self._gamma_step if luma_error > 0 else -self._gamma_step
            new_gamma = min(self._gamma_max, max(self._gamma_min, self._current_gamma + step))
            if new_gamma != self._current_gamma:
                self._current_gamma = new_gamma
                self._gamma_lut = self._build_gamma_lut(self._current_gamma)
        return cv2.LUT(frame, self._gamma_lut)

    @staticmethod
    def _measure_mean_luma(frame):
        """프레임 평균 밝기(0~1) — 채널 평균으로 그레이스케일 변환 생략."""
        b_mean, g_mean, r_mean, _ = cv2.mean(frame)
        return (b_mean + g_mean + r_mean) / 3.0 / 255.0

    @staticmethod
    def _build_gamma_lut(gamma):
        """감마 보정 룩업테이블 — 감마가 바뀔 때만 다시 만들어 재사용(cv2.LUT 자체는
        프레임마다 호출되지만 룩업테이블 재계산은 아니다)."""
        inv_gamma = 1.0 / gamma
        levels = np.arange(256, dtype=np.float32) / 255.0
        return (np.power(levels, inv_gamma) * 255.0).astype(np.uint8)
