"""실행 환경 기록 — 이 실행이 어떤 하드웨어에서 돌았는지 로그에 남긴다 (2026-07-16).

배포 기기가 브랜치(8종)마다 달라서, 실기 로그만 보고 "어느 기기·어느 가속기에서
이 수치가 나왔는지"를 알 수 있어야 튜닝 기록이 증거가 된다. config runtime 절의
타깃 명세(선언값)와 실제 감지값(실측)을 나란히 남긴다 — 둘이 어긋나면 기기 오설치.
카메라 기종·협상 결과는 카메라를 여는 쪽(camera_stream)이 기록한다.
"""
import os
import platform

from src.utils.logger import get_logger

logger = get_logger("utils")


def _physical_memory_gb():
    """장착 RAM(GB). 플랫폼별 API 차이는 실패 시 None으로 흡수한다 (기록용 — 판정 미사용)."""
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        else:
            import ctypes  # 윈도우 — sysconf가 없어 WinAPI로 조회

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total_bytes = status.ullTotalPhys
        return round(total_bytes / 1024 ** 3, 1)
    except Exception:
        return None


def collect_environment(config):
    """실제 실행 환경 스냅샷 dict — 로깅과 분리해 단위 테스트 가능하게 한다."""
    try:
        import onnxruntime as ort  # 무거운 의존 — 사용 시점 임포트

        ort_providers = ort.get_available_providers()
    except Exception:
        ort_providers = []
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": _physical_memory_gb(),
        "ort_providers": ort_providers,   # CUDA/VitisAI 가용 여부가 여기서 드러난다
        "declared_target": config.get("runtime") or {},   # config runtime 절 — 타깃 명세
    }


def log_environment(config):
    """실행 환경(실측)과 타깃 하드웨어 명세(config 선언)를 시작 로그로 남긴다."""
    env = collect_environment(config)
    logger.info(
        "실행 환경(실측): OS=%s · Python=%s · CPU=%s(%s코어) · RAM=%sGB",
        env["os"], env["python"], env["cpu"], env["cpu_cores"],
        "확인불가" if env["ram_gb"] is None else env["ram_gb"],
    )
    logger.info("가속기(ORT providers): %s", ", ".join(env["ort_providers"]) or "없음")
    declared = env["declared_target"]
    if declared:
        logger.info(
            "타깃 명세(config runtime): os=%s · gpu=%s · python=%s",
            declared.get("target_os"), declared.get("target_gpu"), declared.get("python_version"),
        )
    hardware = declared.get("hardware")
    if hardware:
        logger.info("타깃 하드웨어 명세(config): %s", hardware)
    return env
