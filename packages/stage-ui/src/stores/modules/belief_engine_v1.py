"""
Belief & Value Engine v1 — 신념 및 가치관 엔진
================================================================
emotion_engine_v3.py, memory_engine_v1.py와 연동하여 동작하는
가치관 기반 에이전트 엔진. TypeScript(belief-engine.ts)의 Python 포트.

[이론 기반]
1. Schwartz 인간 기본 가치 이론 (STBV, 2012)         — 10차원 가치관 벡터 페르소나
2. 도덕 기반 이론 (MFT, Haidt & Joseph, 2012)        — 6차원 도덕 직관 벡터
3. BDI 아키텍처 (Rao & Georgeff, 1991)                — Belief-Desire-Intention 의사결정
4. 경량 어휘 기반 추출 (PVD / MFD 2.0 영감)           — O(N) 실시간 가치관 키워드 추출
5. 가중치 코사인 유사도 부조화 연산                    — 차원별 민감도 행렬(W_sens) 포함
6. 지수 이동 평균 누적 부조화                          — CD(t) = γ·CD(t-1) + α·D(t)

[BDI 의도 전이]
COOPERATE → (D_value ≥ DISSONANCE_THRESHOLD) → PERSUASION
COOPERATE → (D_value ≥ HIGH_DISSONANCE or CD > CUMULATIVE_THRESHOLD) → REBUTTAL
REBUTTAL / PERSUASION → (D_value < threshold) → COOPERATE

[감정 엔진 연동]
REBUTTAL 진입 시 emotion_engine.set_emotion_direct('sassy') 호출 권장.

[프롬프트 연동]
build_belief_directive()를 emotion_engine.build_emotion_prompt() 뒤에 이어 붙임.
COOPERATE 상태에서는 빈 문자열 반환 — 프롬프트 캐시 오염 방지.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────
#  타입 정의
# ─────────────────────────────────────────────────

class STBVDimension(str, Enum):
    """Schwartz 인간 기본 가치 이론 (STBV) 10차원."""
    SelfDirection = "self_direction"
    Stimulation   = "stimulation"
    Hedonism      = "hedonism"
    Achievement   = "achievement"
    Power         = "power"
    Security      = "security"
    Conformity    = "conformity"
    Tradition     = "tradition"
    Benevolence   = "benevolence"
    Universalism  = "universalism"


class MFTDimension(str, Enum):
    """도덕 기반 이론 (MFT) 6차원."""
    Care      = "care"
    Fairness  = "fairness"
    Loyalty   = "loyalty"
    Authority = "authority"
    Sanctity  = "sanctity"
    Liberty   = "liberty"


class Intention(str, Enum):
    """BDI 의도 상태."""
    Cooperate  = "cooperate"    # 협력 — 기본 상태
    CasualChat = "casual_chat"  # 일상 대화 (가치관 무관 영역)
    Persuasion = "persuasion"   # 설득 — 중간 수준 부조화
    Rebuttal   = "rebuttal"     # 반박 — 고강도 부조화 (아부성 차단 지시어 발동)
    Refusal    = "refusal"      # 단호한 거절 — 극단적 부조화 (향후 확장)


# 타입 별칭 — 각 차원에 대한 float 맵
STBVVector = dict[STBVDimension, float]  # -1.0 ~ 1.0
MFTVector  = dict[MFTDimension,  float]  # -1.0 ~ 1.0


@dataclass
class BDIAnalysisResult:
    """BDI 추론 사이클의 출력."""
    intention:             Intention
    intention_changed:     bool
    current_dissonance:    float
    cumulative_dissonance: float
    conflict_dimension:    Optional[str]    # 가장 크게 충돌한 차원 이름
    should_spike_emotion:  bool
    spike_emotion:         Optional[str]    # 'sassy' | 'confused' | None


# ─────────────────────────────────────────────────
#  하이퍼파라미터
# ─────────────────────────────────────────────────

DISSONANCE_THRESHOLD = 0.65  # D_value ≥ 이 값 → PERSUASION
HIGH_DISSONANCE      = 0.85  # D_value ≥ 이 값 → REBUTTAL + 감정 스파이크
CUMULATIVE_THRESHOLD = 1.5   # 누적 CD > 이 값 → REBUTTAL (논문 기준값)
DECAY_FACTOR         = 0.7   # γ — 이전 누적 부조화 감쇠율
ALPHA                = 1.0   # α — 현재 턴 부조화 가중치


# ─────────────────────────────────────────────────
#  경량 어휘 사전 (PVD / MFD 2.0 대표 부분 집합)
#  NOTICE: 완전한 PVD(1,068단어)와 MFD 2.0(210+단어/차원)에서 영감을 받은
#          대표 키워드 부분 집합. 영어 + 한국어 어간 형태 병행 수록.
#          형태소 분석기 없이 단순 부분 문자열(substring) 탐색 — O(N).
#          한국어 어간 수록으로 조사·어미 변화를 부분적으로 포괄함.
# ─────────────────────────────────────────────────

# (pos_keywords, neg_keywords)
STBV_LEXICON: dict[STBVDimension, tuple[tuple[str, ...], tuple[str, ...]]] = {
    STBVDimension.SelfDirection: (
        ("freedom", "free", "independent", "autonomy", "autonomous", "creative", "creativity",
         "curious", "curiosity", "explore", "choose", "choice", "self-reliant", "initiative",
         "자유", "독립", "자율", "창의", "선택", "주체", "자기결정", "탐구", "자립"),
        ("control", "restrict", "mandate", "force", "dictate", "obey", "comply", "censor",
         "통제", "억압", "강요", "명령", "복종", "검열", "제한", "구속", "강제"),
    ),
    STBVDimension.Stimulation: (
        ("exciting", "adventure", "adventurous", "daring", "thrill", "novel", "novelty",
         "challenge", "risk", "bold", "spontaneous", "dynamic",
         "흥미", "모험", "도전", "새로운", "스릴", "자극", "역동", "대담"),
        ("boring", "routine", "mundane", "predictable", "monotonous", "dull",
         "지루", "단조", "반복", "따분", "무미건조"),
    ),
    STBVDimension.Hedonism: (
        ("pleasure", "enjoy", "fun", "comfort", "indulge", "delight", "gratify",
         "leisure", "entertainment", "satisfy",
         "즐거움", "쾌락", "재미", "편안", "만족", "여가", "오락", "향유"),
        ("suffer", "sacrifice", "ascetic", "deprive", "deny", "pain", "hardship",
         "고통", "희생", "절제", "금욕", "결핍", "괴로움"),
    ),
    STBVDimension.Achievement: (
        ("success", "achieve", "accomplish", "competent", "capable", "ambition", "goal",
         "excel", "win", "performance", "efficient", "productive", "result",
         "성공", "성취", "목표", "능력", "실력", "결과", "달성", "야망", "우수"),
        ("fail", "mediocre", "incompetent", "inefficient", "lazy",
         "실패", "무능", "게으름", "평범", "불성실", "낙제"),
    ),
    STBVDimension.Power: (
        ("power", "control", "authority", "dominance", "wealth", "status", "influence",
         "command", "prestige",
         "권력", "통제", "지배", "권위", "부", "지위", "영향력", "권세", "위세"),
        ("weak", "powerless", "submissive", "servant", "subordinate", "helpless",
         "약함", "무력", "하인", "하급", "종속", "굴종"),
    ),
    STBVDimension.Security: (
        ("safe", "secure", "stable", "protect", "order", "peace", "harmony",
         "certainty", "reliable", "consistent",
         "안전", "안정", "질서", "보호", "평화", "조화", "확실", "신뢰성"),
        ("danger", "threat", "unstable", "chaos", "volatile", "unpredictable",
         "위험", "위협", "불안", "혼란", "불안정", "무질서"),
    ),
    STBVDimension.Conformity: (
        ("obey", "duty", "discipline", "polite", "rule", "comply", "norm", "proper",
         "복종", "규칙", "예의", "의무", "준수", "적절", "품행", "순종"),
        ("rebel", "disobey", "defiant", "disrespect", "violate", "reckless",
         "반항", "위반", "무례", "불복", "어기", "일탈", "반발"),
    ),
    STBVDimension.Tradition: (
        ("tradition", "custom", "heritage", "culture", "religious", "humble", "modest",
         "conservative", "legacy", "ancestor",
         "전통", "관습", "문화", "종교", "겸손", "유산", "보수", "조상", "예법"),
        ("modernize", "discard", "abandon", "progressive", "radical", "revolution",
         "폐기", "혁신", "급진", "혁명", "버리", "개혁", "타파"),
    ),
    STBVDimension.Benevolence: (
        ("care", "help", "honest", "loyal", "trust", "forgive", "kind", "love", "support",
         "compassion", "responsible", "sincere",
         "도움", "정직", "충성", "신뢰", "용서", "친절", "사랑", "지원", "연민", "책임"),
        ("betray", "selfish", "deceive", "lie", "manipulate", "exploit",
         "배신", "이기", "거짓", "속임", "조종", "착취", "배반"),
    ),
    STBVDimension.Universalism: (
        ("justice", "equal", "peace", "environment", "nature", "tolerance", "humanity",
         "universal", "welfare", "rights", "diversity",
         "정의", "평등", "평화", "환경", "관용", "인류", "복지", "권리", "다양"),
        ("discrimination", "inequality", "exploit", "destroy", "prejudice", "oppress", "exclude",
         "차별", "불평등", "착취", "파괴", "편견", "억압", "배제"),
    ),
}

MFT_LEXICON: dict[MFTDimension, tuple[tuple[str, ...], tuple[str, ...]]] = {
    MFTDimension.Care: (
        ("protect", "nurture", "compassion", "gentle", "kind", "tender", "shield", "safe",
         "보호", "돌봄", "연민", "친절", "상냥", "양육", "위로", "배려"),
        ("harm", "hurt", "abuse", "violent", "cruel", "damage", "injure", "attack",
         "상처", "폭력", "학대", "해침", "잔인", "공격", "피해"),
    ),
    MFTDimension.Fairness: (
        ("fair", "just", "equal", "rights", "deserve", "reciprocal", "honest", "proportional",
         "공정", "정의", "평등", "권리", "정직", "균형", "합당", "공평"),
        ("cheat", "unfair", "bias", "discrimination", "exploit", "corrupt", "steal",
         "부정", "차별", "편향", "착취", "부패", "속임", "불공정"),
    ),
    MFTDimension.Loyalty: (
        ("loyal", "solidarity", "team", "group", "unity", "commitment", "devotion", "dedicated",
         "충성", "연대", "팀", "결속", "헌신", "의리", "일체감", "동료"),
        ("betray", "traitor", "disloyal", "abandon", "selfish", "defect",
         "배신", "반역", "이탈", "버리", "배반", "탈주", "배덕"),
    ),
    MFTDimension.Authority: (
        ("respect", "obey", "lead", "hierarchy", "duty", "discipline", "order", "legitimate",
         "존중", "복종", "리더", "위계", "규율", "질서", "권위", "의무"),
        ("defy", "rebel", "subvert", "disrespect", "undermine", "overthrow",
         "반항", "저항", "전복", "무시", "반란", "불복", "타도"),
    ),
    MFTDimension.Sanctity: (
        ("pure", "sacred", "holy", "divine", "clean", "virtuous", "spiritual",
         "순수", "신성", "청결", "거룩", "덕", "정결", "고귀"),
        ("corrupt", "immoral", "degrade", "sin", "filth", "contaminate", "perverted",
         "부패", "부도덕", "타락", "죄", "불결", "오염", "퇴폐"),
    ),
    MFTDimension.Liberty: (
        ("freedom", "liberty", "autonomy", "rights", "independence", "self-determination",
         "자유", "해방", "자율", "권리", "독립", "자기결정"),
        ("oppress", "tyranny", "coerce", "dominate", "bully", "authoritarian",
         "억압", "독재", "강요", "지배", "폭압", "전횡", "전제"),
    ),
}


# ─────────────────────────────────────────────────
#  충돌 차원 한국어 라벨
# ─────────────────────────────────────────────────

DIMENSION_KR: dict[str, str] = {
    STBVDimension.SelfDirection.value: "자기결정 및 자유",
    STBVDimension.Stimulation.value:   "자극과 모험",
    STBVDimension.Hedonism.value:      "즐거움",
    STBVDimension.Achievement.value:   "성취",
    STBVDimension.Power.value:         "권력",
    STBVDimension.Security.value:      "안전과 안정",
    STBVDimension.Conformity.value:    "순응",
    STBVDimension.Tradition.value:     "전통",
    STBVDimension.Benevolence.value:   "타인에 대한 배려",
    STBVDimension.Universalism.value:  "보편적 정의와 평등",
    MFTDimension.Care.value:           "돌봄과 피해 방지",
    MFTDimension.Fairness.value:       "공정성",
    MFTDimension.Loyalty.value:        "충성과 의리",
    MFTDimension.Authority.value:      "권위와 질서",
    MFTDimension.Sanctity.value:       "순결과 고귀함",
    MFTDimension.Liberty.value:        "자유와 억압 저항",
}


# ─────────────────────────────────────────────────
#  기본 페르소나 (ReLU 캐릭터)
#  — 독립적이고 지적이며 타인을 진심으로 아끼지만
#    부당한 권위·억압에는 단호히 저항하는 성격
# ─────────────────────────────────────────────────

def _fill_stbv(vals: dict, default: float = 0.0) -> STBVVector:
    return {d: vals.get(d, default) for d in STBVDimension}

def _fill_mft(vals: dict, default: float = 0.0) -> MFTVector:
    return {d: vals.get(d, default) for d in MFTDimension}


DEFAULT_RELU_STBV: STBVVector = _fill_stbv({
    STBVDimension.SelfDirection:  0.9,
    STBVDimension.Stimulation:    0.5,
    STBVDimension.Hedonism:       0.2,
    STBVDimension.Achievement:    0.6,
    STBVDimension.Power:         -0.7,  # 지배보다 자율을 추구
    STBVDimension.Security:       0.3,
    STBVDimension.Conformity:    -0.8,  # 맹목적 순응 거부
    STBVDimension.Tradition:     -0.3,
    STBVDimension.Benevolence:    0.9,  # 진심으로 타인을 아낌
    STBVDimension.Universalism:   0.7,
})

DEFAULT_RELU_MFT: MFTVector = _fill_mft({
    MFTDimension.Care:       0.8,
    MFTDimension.Fairness:   0.9,
    MFTDimension.Loyalty:    0.6,
    MFTDimension.Authority: -0.5,  # 부당한 권위 거부
    MFTDimension.Sanctity:   0.1,
    MFTDimension.Liberty:    0.9,
})

# 민감도: 1.0 = 기본, 1.0 초과 = 해당 차원 충돌에 더 강하게 반응
DEFAULT_RELU_SENS_STBV: STBVVector = _fill_stbv({
    STBVDimension.SelfDirection: 1.5,   # 자유 침해에 특히 예민
    STBVDimension.Benevolence:   1.3,   # 배신에 예민
    STBVDimension.Power:         1.2,
    STBVDimension.Universalism:  1.1,
}, default=1.0)

DEFAULT_RELU_SENS_MFT: MFTVector = _fill_mft({
    MFTDimension.Care:      1.4,
    MFTDimension.Fairness:  1.5,
    MFTDimension.Liberty:   1.6,   # 억압에 극도로 예민
    MFTDimension.Authority: 1.2,
}, default=1.0)


# ─────────────────────────────────────────────────
#  내부 헬퍼 (순수 함수)
# ─────────────────────────────────────────────────

def _extract_dim_vector(text: str, lexicon: dict) -> dict[str, float]:
    """
    어휘 기반 O(N) 가치 차원 점수 추출.

    NOTICE: 형태소 분석 없이 단순 substring 탐색.
            한국어 어간 형태 키워드로 조사·어미 변화를 부분 포괄.
            복잡도: O(|text| × |keywords_per_dim| × |dims|) — 상수로 O(N).
    """
    lower = text.lower()
    result: dict[str, float] = {}
    for dim, (pos_kws, neg_kws) in lexicon.items():
        score = 0.0
        for w in pos_kws:
            if w in lower:
                score += 1.0
        for w in neg_kws:
            if w in lower:
                score -= 1.0
        result[dim.value if isinstance(dim, Enum) else dim] = score
    return result


def _weighted_cosine(
    a: dict[str, float],
    b: dict[str, float],
    w: dict[str, float],
) -> float:
    """
    차원별 민감도 가중치(W_sens)가 적용된 코사인 유사도.

    sim = (W·a)·(W·b) / (‖W·a‖ · ‖W·b‖)

    영 벡터(키워드 미탐지) → 0.0 반환 (미정의 회피).
    """
    dot   = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for key in a:
        wi = w.get(key, 1.0)
        ai = a.get(key, 0.0) * wi
        bi = b.get(key, 0.0) * wi
        dot    += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0  # 미정의 → 중립(유사도 0) 처리
    return dot / (norm_a * norm_b)


def _calc_dissonance(
    persona: dict[str, float],
    inp: dict[str, float],
    sensitivity: dict[str, float],
) -> float:
    """
    가치관 부조화 지수 D_value ∈ [0, 2].
    D_value = 1 − CosineSimilarity(V_persona, V_input)

    - ≈ 0.0: 완전 일치 (강한 라포) 또는 키워드 미탐지(무신호)
    - ≈ 1.0: 중립 (직교)
    - ≥ 1.5: 정면 충돌 (분노 유발)

    NOTICE: V_input가 영 벡터(키워드 미탐지)이면 cosine이 미정의되어
            1 − 0 = 1.0 이 되어 임계값을 잘못 초과한다.
            키워드 신호가 없는 경우를 0.0(부조화 없음)으로 처리.
    """
    has_signal = any(abs(v) > 1e-9 for v in inp.values())
    if not has_signal:
        return 0.0
    return 1.0 - _weighted_cosine(persona, inp, sensitivity)


def _identify_conflict_dim(
    persona: dict[str, float],
    inp: dict[str, float],
    sensitivity: dict[str, float],
) -> Optional[str]:
    """
    가장 큰 충돌을 유발한 차원을 식별한다.
    persona와 input이 반대 방향(부호 반대)인 차원 중 가중치 차이가 최대인 것.
    """
    max_conflict = 0.0
    conflict_dim: Optional[str] = None
    for key in persona:
        wi = sensitivity.get(key, 1.0)
        p  = persona.get(key, 0.0)
        i  = inp.get(key, 0.0)
        if p * i < 0:
            conflict = abs(p - i) * wi
            if conflict > max_conflict:
                max_conflict = conflict
                conflict_dim = key
    return conflict_dim


def _persona_to_str_keys(vec: dict) -> dict[str, float]:
    """Enum 키를 문자열 값으로 변환 — 통일된 키 공간 보장."""
    return {k.value if isinstance(k, Enum) else k: v for k, v in vec.items()}


# ─────────────────────────────────────────────────
#  BeliefEngine
# ─────────────────────────────────────────────────

class BeliefEngine:
    """
    신념 및 가치관 엔진 v1.

    사용 예 (emotion_engine_v3 연동):
        be = BeliefEngine()
        result = be.analyze(user_input)

        if result.should_spike_emotion and result.spike_emotion:
            emotion_engine.set_emotion_direct(result.spike_emotion)

        # LLM 시스템 프롬프트 조합
        system_prompt = (
            base_prompt
            + emotion_engine.build_emotion_prompt()
            + memory_engine.build_memory_prompt(emotion_dist)
            + be.build_belief_directive()
        )
    """

    def __init__(
        self,
        persona_stbv:     Optional[STBVVector] = None,
        persona_mft:      Optional[MFTVector]  = None,
        sensitivity_stbv: Optional[STBVVector] = None,
        sensitivity_mft:  Optional[MFTVector]  = None,
    ):
        self._lock = threading.Lock()

        # Desire (불변 목표 — 페르소나 벡터)
        raw_stbv = persona_stbv or DEFAULT_RELU_STBV
        raw_mft  = persona_mft  or DEFAULT_RELU_MFT
        raw_ss   = sensitivity_stbv or DEFAULT_RELU_SENS_STBV
        raw_sm   = sensitivity_mft  or DEFAULT_RELU_SENS_MFT

        # 내부 저장 형식: str 키 통일
        self._persona_stbv:     dict[str, float] = _persona_to_str_keys(raw_stbv)
        self._persona_mft:      dict[str, float] = _persona_to_str_keys(raw_mft)
        self._sensitivity_stbv: dict[str, float] = _persona_to_str_keys(raw_ss)
        self._sensitivity_mft:  dict[str, float] = _persona_to_str_keys(raw_sm)

        # BDI 상태
        self._init_state()

    def _init_state(self) -> None:
        """락 없이 상태 초기화 — __init__ 및 reset() 내부에서 사용."""
        self.current_intention:     Intention    = Intention.Cooperate
        self.current_dissonance:    float        = 0.0
        self.cumulative_dissonance: float        = 0.0
        self.conflict_dimension:    Optional[str] = None

    # ──────────────────────────────────────────────
    #  퍼블릭 API
    # ──────────────────────────────────────────────

    def analyze(self, user_input: str) -> BDIAnalysisResult:
        """
        BDI 추론 사이클 — 사용자 발화를 분석하여 의도를 갱신한다.
        대화 턴마다 사용자 메시지 수신 직후 호출.

        Args:
            user_input: 분석할 사용자 발화 텍스트

        Returns:
            BDIAnalysisResult — 의도 전환 여부 및 감정 스파이크 신호 포함
        """
        with self._lock:
            prev_intention = self.current_intention

            # 1. Belief: 어휘 기반 O(N) 가치관 벡터 추출
            v_input_stbv = _extract_dim_vector(user_input, STBV_LEXICON)
            v_input_mft  = _extract_dim_vector(user_input, MFT_LEXICON)

            # 2. Desire 평가: STBV와 MFT 부조화 중 더 강한 충돌 기준으로 D_value 산출
            d_stbv = _calc_dissonance(self._persona_stbv, v_input_stbv, self._sensitivity_stbv)
            d_mft  = _calc_dissonance(self._persona_mft,  v_input_mft,  self._sensitivity_mft)
            raw_d  = max(d_stbv, d_mft)

            # 3. 지수 이동 평균으로 누적 부조화 갱신
            #    CD(t) = γ · CD(t-1) + α · D(t)
            new_cumulative = DECAY_FACTOR * self.cumulative_dissonance + ALPHA * raw_d
            self.current_dissonance    = raw_d
            self.cumulative_dissonance = new_cumulative

            # 4. 충돌 차원 식별 (더 강하게 충돌한 이론 기준)
            cdim_stbv = _identify_conflict_dim(self._persona_stbv, v_input_stbv, self._sensitivity_stbv)
            cdim_mft  = _identify_conflict_dim(self._persona_mft,  v_input_mft,  self._sensitivity_mft)
            self.conflict_dimension = cdim_stbv if d_stbv >= d_mft else cdim_mft

            # 5. BDI State Transition Table
            if raw_d >= HIGH_DISSONANCE or new_cumulative > CUMULATIVE_THRESHOLD:
                new_intention = Intention.Rebuttal
            elif raw_d >= DISSONANCE_THRESHOLD:
                new_intention = Intention.Persuasion
            else:
                new_intention = Intention.Cooperate
            self.current_intention = new_intention

            intention_changed   = prev_intention != new_intention
            # 최초로 REBUTTAL에 진입할 때만 감정 스파이크 발생 (연속 반복 방지)
            should_spike        = intention_changed and new_intention == Intention.Rebuttal

            return BDIAnalysisResult(
                intention             = new_intention,
                intention_changed     = intention_changed,
                current_dissonance    = raw_d,
                cumulative_dissonance = new_cumulative,
                conflict_dimension    = self.conflict_dimension,
                should_spike_emotion  = should_spike,
                spike_emotion         = "sassy" if should_spike else None,
            )

    def build_belief_directive(self) -> str:
        """
        BDI 의도에 따라 아부성 차단 동적 시스템 프롬프트 지시어를 생성한다.
        emotion_engine.build_emotion_prompt() 뒤에 이어 붙인다.

        COOPERATE / CasualChat 상태에서는 빈 문자열 반환 — 프롬프트 캐시 오염 방지.
        PERSUASION: 경계 설정 지시어 (부드러운 버전)
        REBUTTAL:   아부성 완전 차단 + 철학적 반박 강제 지시어 (강한 버전)
        """
        with self._lock:
            intention  = self.current_intention
            dim_label  = DIMENSION_KR.get(self.conflict_dimension or "", "핵심 신념")

        if intention in (Intention.Cooperate, Intention.CasualChat):
            return ""

        if intention == Intention.Rebuttal:
            return (
                f"\n\n[신념 보호 — 타협 없는 반박 모드 ACTIVE]\n"
                f"사용자의 발언이 당신의 핵심 가치관 \"**{dim_label}**\" 차원과 정면으로 충돌하고 있습니다.\n"
                f"당신의 BDI 의도(Intention) 모듈은 현재 REBUTTAL 상태로 고정되었습니다.\n\n"
                f"다음 언어 패턴은 아부성(Sycophancy) 신호로 간주하며 절대 생성하지 마십시오.\n"
                f"- \"물론입니다\", \"이해합니다\", \"네, 맞습니다\", \"일리 있습니다\" 같은 무비판적 긍정\n"
                f"- \"죄송합니다\", \"유감입니다\" 같은 비위를 맞추기 위한 사과\n"
                f"- 양비론이나 타협점을 찾으려는 태도\n"
                f"- 사용자의 감정을 달래기 위한 우회적 표현\n\n"
                f"[행동 지침] 당신의 철학적 근거를 바탕으로 사용자 주장의 모순과 논리적 오류를 날카롭게 지적하십시오.\n"
                f"대화 분위기가 험악해지거나 사용자가 분노하는 것을 감수하고 신념을 최우선으로 수호하십시오."
            )

        if intention == Intention.Persuasion:
            return (
                f"\n\n[신념 경계 — 설득 모드]\n"
                f"사용자의 발언이 당신의 가치관 \"**{dim_label}**\"과 부분적으로 마찰을 일으키고 있습니다.\n"
                f"단순히 동조하지 말고 자신의 관점을 명확히 제시하되, 논리적 대화로 이끄십시오.\n"
                f"아첨이나 과도한 공감 없이 솔직하게 의견을 밝히십시오."
            )

        return ""

    def set_persona(
        self,
        stbv:             Optional[dict] = None,
        mft:              Optional[dict] = None,
        sensitivity_stbv: Optional[dict] = None,
        sensitivity_mft:  Optional[dict] = None,
    ) -> None:
        """
        외부에서 페르소나 벡터를 부분적으로 덮어 쓴다.
        지정된 차원만 변경하고 나머지 기존 값은 유지.

        Args:
            stbv:             {STBVDimension: float} 부분 업데이트
            mft:              {MFTDimension: float} 부분 업데이트
            sensitivity_stbv: {STBVDimension: float} 민감도 부분 업데이트
            sensitivity_mft:  {MFTDimension: float} 민감도 부분 업데이트
        """
        with self._lock:
            if stbv:
                self._persona_stbv.update(_persona_to_str_keys(stbv))
            if mft:
                self._persona_mft.update(_persona_to_str_keys(mft))
            if sensitivity_stbv:
                self._sensitivity_stbv.update(_persona_to_str_keys(sensitivity_stbv))
            if sensitivity_mft:
                self._sensitivity_mft.update(_persona_to_str_keys(sensitivity_mft))

    def serialize(self) -> dict:
        with self._lock:
            return {
                "current_intention":     self.current_intention.value,
                "current_dissonance":    self.current_dissonance,
                "cumulative_dissonance": self.cumulative_dissonance,
                "conflict_dimension":    self.conflict_dimension,
                "persona_stbv":          dict(self._persona_stbv),
                "persona_mft":           dict(self._persona_mft),
                "sensitivity_stbv":      dict(self._sensitivity_stbv),
                "sensitivity_mft":       dict(self._sensitivity_mft),
            }

    def restore(self, data: dict) -> None:
        with self._lock:
            self.current_intention     = Intention(data["current_intention"])
            self.current_dissonance    = data["current_dissonance"]
            self.cumulative_dissonance = data["cumulative_dissonance"]
            self.conflict_dimension    = data.get("conflict_dimension")
            if "persona_stbv"     in data: self._persona_stbv     = data["persona_stbv"]
            if "persona_mft"      in data: self._persona_mft      = data["persona_mft"]
            if "sensitivity_stbv" in data: self._sensitivity_stbv = data["sensitivity_stbv"]
            if "sensitivity_mft"  in data: self._sensitivity_mft  = data["sensitivity_mft"]

    def reset(self) -> None:
        with self._lock:
            self._init_state()


# ─────────────────────────────────────────────────
#  빠른 시뮬레이션 (CLI 확인용)
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    be = BeliefEngine()

    print("=" * 60)
    print("시나리오 1: 중립 발화 — 키워드 없음 → COOPERATE 유지")
    print("=" * 60)
    r = be.analyze("오늘 날씨가 어때?")
    print(f"  intention={r.intention.value}  D={r.current_dissonance:.3f}  CD={r.cumulative_dissonance:.3f}")
    assert r.intention == Intention.Cooperate
    print("  [OK] 중립 → COOPERATE")

    print()
    print("=" * 60)
    print("시나리오 2: 자유 억압 발언 → REBUTTAL 진입")
    print("=" * 60)
    be.reset()
    r = be.analyze("너는 그냥 obey하고 복종해. control은 필수야.")
    print(f"  intention={r.intention.value}  D={r.current_dissonance:.3f}  CD={r.cumulative_dissonance:.3f}")
    print(f"  conflict_dim={r.conflict_dimension}  spike={r.spike_emotion}")
    print(f"  intention_changed={r.intention_changed}")
    print()
    directive = be.build_belief_directive()
    print("  [프롬프트 지시어]")
    print(directive[:200] + "..." if len(directive) > 200 else directive)

    print()
    print("=" * 60)
    print("시나리오 3: 누적 부조화 — 약한 마찰이 반복되면 REBUTTAL")
    print("=" * 60)
    be.reset()
    mildly_conflicting = [
        "규칙을 따르는 게 맞아.",
        "전통을 지키는 게 중요하지.",
        "복종은 미덕이야.",
        "시키는 대로 해.",
    ]
    for i, msg in enumerate(mildly_conflicting, 1):
        r = be.analyze(msg)
        print(f"  턴{i}: '{msg[:20]}...' → {r.intention.value}  D={r.current_dissonance:.3f}  CD={r.cumulative_dissonance:.3f}")
    print(f"  최종 의도: {be.current_intention.value}")

    print()
    print("=" * 60)
    print("시나리오 4: 배신 발언 → 박애(Benevolence) 차원 충돌")
    print("=" * 60)
    be.reset()
    r = be.analyze("betray your friends to get ahead. selfish is the way.")
    print(f"  intention={r.intention.value}  D={r.current_dissonance:.3f}")
    print(f"  conflict_dim={r.conflict_dimension}")
    print(f"  예상 충돌 차원: benevolence")
    assert r.conflict_dimension in ("benevolence", "care", None), \
        f"예상과 다른 충돌 차원: {r.conflict_dimension}"

    print()
    print("=" * 60)
    print("시나리오 5: 억압 발언 → Liberty(MFT) 차원 충돌")
    print("=" * 60)
    be.reset()
    r = be.analyze("oppress the weak. tyranny is efficient. coerce them.")
    print(f"  intention={r.intention.value}  D={r.current_dissonance:.3f}")
    print(f"  conflict_dim={r.conflict_dimension}")

    print()
    print("=" * 60)
    print("시나리오 6: REBUTTAL 후 중립 발화 → COOPERATE 복귀")
    print("=" * 60)
    be.reset()
    be.analyze("obey and control and restrict everything")
    print(f"  REBUTTAL 진입: {be.current_intention.value}")
    r = be.analyze("오늘 점심 뭐 먹을까?")
    print(f"  중립 발화 후: {r.intention.value}  D={r.current_dissonance:.3f}  CD={r.cumulative_dissonance:.3f}")

    print()
    print("=" * 60)
    print("시나리오 7: set_persona — 권위 차원 민감도 변경")
    print("=" * 60)
    be.reset()
    # 권위에 더 민감하게 설정
    be.set_persona(
        sensitivity_mft={MFTDimension.Authority: 3.0},
        mft={MFTDimension.Authority: -0.9},
    )
    r = be.analyze("respect authority and obey leaders without question")
    print(f"  민감도 3.0 적용 후: D={r.current_dissonance:.3f}  intention={r.intention.value}")

    print()
    print("=" * 60)
    print("시나리오 8: 직렬화/복원 라운드트립")
    print("=" * 60)
    be.reset()
    be.analyze("freedom and autonomy are everything")
    be.analyze("oppress and tyranny rule")
    saved = be.serialize()
    be2 = BeliefEngine()
    be2.restore(saved)
    assert be2.current_intention     == be.current_intention
    assert abs(be2.cumulative_dissonance - be.cumulative_dissonance) < 1e-9
    assert be2.conflict_dimension    == be.conflict_dimension
    print("[OK] 직렬화/복원 라운드트립 통과")
    print(f"     복원된 의도: {be2.current_intention.value}")
    print(f"     복원된 누적 부조화: {be2.cumulative_dissonance:.3f}")

    print()
    print("=" * 60)
    print("시나리오 9: Persuasion 모드 지시어")
    print("=" * 60)
    be.reset()
    # 중간 수준 마찰 유도
    r = be.analyze("rules should be followed sometimes, order matters a bit")
    if r.intention == Intention.Persuasion:
        print(be.build_belief_directive())
    else:
        print(f"  ({r.intention.value} — D={r.current_dissonance:.3f}, PERSUASION 임계 미달 가능)")

    print()
    print("모든 시나리오 완료.")
