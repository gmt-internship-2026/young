"""ocr 모듈 — 주민등록증에서 이름·주민등록번호를 인식해 입력칸 자동 채움 이벤트를 만든다.

요구사항(2026-07-10): 주민등록증을 카메라에 보여주면 주민번호·이름이 자동으로
입력칸에 들어간다. UI가 본인확인 화면에 진입할 때 OCR 모드를 요청하고(demo_server
POST /ocr/start), 인식이 성공하면 fill_id_fields 이벤트로 결과가 전달된다.

개인정보 원칙 (개인정보보호법 — 주민등록번호 처리의 법적 근거는 회사 확인 필요, TODO №11):
- 카메라 프레임과 인식 결과를 디스크에 저장하지 않는다 (전 과정 메모리 처리)
- 로그에는 마스킹된 값만 남긴다: 홍*동 / 950101-1******
- 이벤트 payload에는 원문이 실린다 — 수신한 UI(회사 프로그램)의 처리 책임 범위

파싱 함수(parse_id_fields 등)는 easyocr 없이 동작하는 순수 로직이라
단위 테스트로 검증한다 (tests/test_idcard_parse.py).
"""
import re

from src.utils.logger import get_logger

logger = get_logger("ocr")

# 주민등록번호: 생년월일 6자리 - 뒷자리 7자리(첫 자리 1~8). OCR 특성상 공백·대시 변형 허용
RRN_PATTERN = re.compile(r"(\d{6})\s*[-—–‐]?\s*([1-8]\d{6})")
NAME_PATTERN = re.compile(r"^[가-힣]{2,5}$")
# 이름으로 오인하기 쉬운 서식 단어들 (주민등록증 표면의 다른 한글 텍스트)
NAME_STOPWORDS = {"주민등록증", "주민", "등록증", "성명", "주소", "발급일", "시장", "도지사",
                  "군수", "구청장", "특별시장", "광역시장", "제주특별자치도지사"}

RRN_CHECK_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)


def validate_rrn_checksum(rrn_digits):
    """주민등록번호 검증 공식. 참고용 신호로만 쓴다 — 2020년 10월 이후 발급분은
    뒷자리가 임의번호라 이 공식을 따르지 않을 수 있다 (불일치해도 기각하지 않는다)."""
    if len(rrn_digits) != 13 or not rrn_digits.isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(rrn_digits[:12], RRN_CHECK_WEIGHTS))
    return (11 - total % 11) % 10 == int(rrn_digits[12])


def mask_name(name):
    """홍길동 -> 홍*동, 김구 -> 김* (로그·화면 표시용)."""
    if len(name) <= 1:
        return "*"
    if len(name) == 2:
        return name[0] + "*"
    return name[0] + "*" * (len(name) - 2) + name[-1]


def mask_rrn(front6, back7):
    """950101-1234567 -> 950101-1****** (로그·화면 표시용)."""
    return f"{front6}-{back7[0]}{'*' * 6}"


def _clean_name_candidate(text):
    """괄호(한자 병기)·공백을 걷어내고 한글 이름 후보만 남긴다."""
    text = re.sub(r"[(（].*?[)）]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def parse_id_fields(text_items, min_conf):
    """OCR 결과 [(text, conf), ...] (위->아래 순)에서 이름·주민번호를 뽑는다.

    반환: {"name": str, "rrn": str, "conf": float} 또는 None (둘 중 하나라도 못 찾으면).
    이름은 주민번호 줄보다 위에서, 서식 단어를 제외한 한글 2~5자를 고른다.
    """
    rrn = None
    rrn_conf = 0.0
    rrn_index = None
    for index, (text, conf) in enumerate(text_items):
        if conf < min_conf:
            continue
        match = RRN_PATTERN.search(text.replace(" ", ""))
        if match is None:
            match = RRN_PATTERN.search(text)
        if match is not None:
            front6, back7 = match.group(1), match.group(2)
            rrn = f"{front6}-{back7}"
            rrn_conf = conf
            rrn_index = index
            if not validate_rrn_checksum(front6 + back7):
                logger.info("주민번호 검증 공식 불일치 — 2020.10 이후 발급분일 수 있어 계속 진행")
            break
    if rrn is None:
        return None

    name = None
    name_conf = 0.0
    # 주민번호 줄에서 가까운 위쪽부터 이름 후보를 찾는다 (증 표면에서 이름이 번호 바로 위)
    for text, conf in reversed(text_items[:rrn_index]):
        if conf < min_conf:
            continue
        candidate = _clean_name_candidate(text)
        if candidate in NAME_STOPWORDS:
            continue
        if NAME_PATTERN.match(candidate):
            name = candidate
            name_conf = conf
            break
    if name is None:
        return None

    front6, back7 = rrn.split("-")
    logger.info("주민등록증 인식: %s / %s", mask_name(name), mask_rrn(front6, back7))
    return {"name": name, "rrn": rrn, "conf": min(rrn_conf, name_conf)}


class IdCardReader:
    """EasyOCR 기반 주민등록증 판독기. read(frame) -> fields dict | None.

    easyocr 로딩(모델 다운로드 포함)이 무거워 첫 read 시점에 초기화한다.
    파이프라인은 이 클래스를 OCR 워커 스레드에서만 호출한다.
    """

    def __init__(self, config):
        ocr = config["ocr"]
        self._languages = ocr["languages"]
        self._use_gpu = ocr["gpu"]
        self._min_text_conf = ocr["min_text_conf"]
        self._guide_region_ratio = ocr["guide_region_ratio"]
        self._reader = None

    def _ensure_reader(self):
        if self._reader is not None:
            return
        import easyocr  # 무거운 의존성 — OCR을 끈 환경에서는 임포트조차 하지 않는다

        try:
            self._reader = easyocr.Reader(self._languages, gpu=self._use_gpu)
        except Exception:  # GPU 미지원(개발 PC 등)이면 CPU로 폴백
            logger.warning("EasyOCR GPU 초기화 실패 — CPU로 폴백합니다")
            self._reader = easyocr.Reader(self._languages, gpu=False)
        logger.info("EasyOCR 로딩 완료 (langs=%s)", self._languages)

    def _crop_guide_region(self, frame):
        h_px, w_px = frame.shape[:2]
        x1_r, y1_r, x2_r, y2_r = self._guide_region_ratio
        return frame[int(h_px * y1_r):int(h_px * y2_r), int(w_px * x1_r):int(w_px * x2_r)]

    def read(self, frame):
        """원본(반전 없는) 프레임에서 이름·주민번호를 인식한다. 실패 시 None."""
        self._ensure_reader()
        crop = self._crop_guide_region(frame)
        # readtext 결과: [(bbox, text, conf), ...] — 대체로 위에서 아래 순서
        results = self._reader.readtext(crop)
        text_items = [(text, float(conf)) for _, text, conf in results]
        if not text_items:
            return None
        return parse_id_fields(text_items, self._min_text_conf)
