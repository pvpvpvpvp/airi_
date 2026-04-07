"""
Emotion Engine v3 — 9-theory cognitive science integration
===========================================================
TypeScript(emotion-engine.ts)의 Python 포트. 레드팀 검토 후 수정 반영본.

[이론 구성]
1. 베이즈 뇌 (Bayesian Brain)         — 확률적 감정 업데이트
2. 드리프트 확산 모델 (DDM)            — 결정 지연과 오류
3. 제한된 합리성 (Bounded Rationality) — 만족화 노이즈
4. 마르코프 연쇄 (Markov Chain)        — 감정의 관성
5. 쾌락 적응 (Hedonic Adaptation)     — 시간/회차에 따른 감정 소멸
6. 평가 이론 (Appraisal Theory)       — 상태 의존적 해석
7. 피크엔드 법칙 (Peak-End Rule)      — 강렬한 기억의 편향 (recency 제한 추가)
8. 자아 고갈 (Ego Depletion)          — 감정 에너지 소모
9. 인지 부조화 (Cognitive Dissonance)  — 감정 충돌 시 불안정

[v3 변경 사항 — 레드팀 수정]
- FIX: decay가 3턴마다 1회 실행되도록 counter 리셋 추가
- FIX: intensityFactor = max(0.3, maxProb) — 지배적 감정일수록 decay 강하게
- FIX: DDM winner 체크를 step 완료 후 argmax로 — 순서 편향 제거
- FIX: recentEvents 윈도우 4→5로 통일
- FIX: processEvent에 unknown 이벤트 가드 추가
- FIX: conversationTurnCount 도입 — 양쪽 경로(processEvent/set_emotion_direct) 모두 증가
- FIX: peakDesc 15턴 recency 제한 — 오래된 피크는 프롬프트에서 자동 소멸
- FIX: peakIntensity dead state 제거
- FIX: 이벤트 지시 프롬프트를 정적 상수로 분리 (매 턴 재생성 방지)
"""

from __future__ import annotations

import dataclasses
import math
import random
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────
#  PRNG — xoshiro128+ 시드 기반 난수 (재현성 보장)
# ─────────────────────────────────────────────────

class SeededRandom:
    """
    xoshiro128+ — splitmix32로 시드를 4-state로 확장 후 난수 생성.
    주기: 2^128 - 1.
    """
    def __init__(self, seed: Optional[int] = None):
        import time
        self._s = [0, 0, 0, 0]
        self.seed(seed if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF)

    def seed(self, seed: int) -> None:
        z = seed & 0xFFFFFFFF
        def splitmix() -> int:
            nonlocal z
            z = (z + 0x9E3779B9) & 0xFFFFFFFF
            t = z ^ (z >> 16)
            t = (t * 0x21F0AAAD) & 0xFFFFFFFF
            t = t ^ (t >> 15)
            t = (t * 0x735A2D97) & 0xFFFFFFFF
            return (t ^ (t >> 15)) & 0xFFFFFFFF
        self._s = [splitmix(), splitmix(), splitmix(), splitmix()]

    def random(self) -> float:
        s = self._s
        result = (s[0] + s[3]) & 0xFFFFFFFF
        t = (s[1] << 9) & 0xFFFFFFFF
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = ((s[3] << 11) | (s[3] >> 21)) & 0xFFFFFFFF
        return result / 4294967296.0

    def get_state(self) -> tuple[int, int, int, int]:
        return tuple(self._s)  # type: ignore[return-value]

    def set_state(self, state: tuple[int, int, int, int]) -> None:
        self._s = list(state)


# ─────────────────────────────────────────────────
#  감정 & 이벤트 정의
# ─────────────────────────────────────────────────

class Emotion(str, Enum):
    Happy   = "happy"
    Neutral = "neutral"
    Sassy   = "sassy"
    Tired   = "tired"
    Excited = "excited"
    Confused = "confused"

EMOTIONS = list(Emotion)

EMOTION_KR: dict[Emotion, str] = {
    Emotion.Happy:   "기분좋음",
    Emotion.Neutral: "평온함",
    Emotion.Sassy:   "깐찐함",
    Emotion.Tired:   "귀찮음",
    Emotion.Excited: "설렘",
    Emotion.Confused:"혼란",
}

EMOTION_DIRECTIVES: dict[Emotion, str] = {
    Emotion.Happy:   "지금 기분이 좋다. 밝고 긍정적인 톤으로 대답하고, ㅎㅎ 같은 표현을 자연스럽게 쓴다.",
    Emotion.Neutral: "지금 평온한 상태다. 담담하고 차분하게 대답한다.",
    Emotion.Sassy:   "지금 좀 깐찐하다. 현명스럽고 아이러니하게 대답하되, 본질적으로는 도움을 주려 한다. '흠...' 같은 표현을 쓸 수 있다.",
    Emotion.Tired:   "지금 귀찮고 피곤하다. 대답을 짧게 하려 하고, '대충', '나중에' 같은 표현을 쓴다.",
    Emotion.Excited: "지금 매우 설렌다. 대화에 '대박', '헐' 같은 감탄사를 자연스럽게 쓴다.",
    Emotion.Confused:"지금 감정적으로 혼란스럽다. 말이 약간 어수선하고, '잠깐만', '뭔지', '아닌가...' 같은 표현을 쓴다.",
}

EMOTION_FILLERS: dict[Emotion, dict[str, list[str]]] = {
    Emotion.Happy: {
        "short":      ["ㅎㅎ", "음~", "잠깐~", "잠깐!", "정말다", "히히"],
        "long":       ["음~ 잠깐 생각 좀 해볼게 ㅎㅎ", "음음 잠깐! 좋은 생각 났어",
                       "아 이거 재밌는데, 잠깐~", "ㅎㅎ 기다려봐 금방 말해줄게", "잠깐 이거 좀 생각해볼게!"],
        "energy_low": ["ㅎ... 잠깐만", "음 음... 기다려봐"],
    },
    Emotion.Neutral: {
        "short":      ["음...", "잠깐만", "어디 보자", "그게...", "음,"],
        "long":       ["음... 잠깐만 생각 좀 해볼게", "어디 보자... 잠깐",
                       "그게 맞이야... 잠깐만", "음, 잠깐. 정리 좀 하고", "음 이거 좀 생각해볼게"],
        "energy_low": ["음......", "잠깐..."],
    },
    Emotion.Sassy: {
        "short":      ["흠...", "음 뭔", "어휴", "쯧", "아니", "뭔데"],
        "long":       ["흠... 이걸 내가 왜 생각해야 되는 건데", "음 잠깐만 좀. 생각 중이에요",
                       "어휴... 기다려봐", "모르겠어 잠깐만 좀", "쯧... 이것도 내가 해야 돼?", "아니 잠깐만 좀 기다려"],
        "energy_low": ["흠...... 진짜", "모르겠어... 잠깐...", "어휴......"],
    },
    Emotion.Tired: {
        "short":      ["음...", "일...", "으...", "하아", "어..."],
        "long":       ["음... 잠깐만 좀 쉬고", "일... 이거 지금 꼭 해야 돼?",
                       "으 잠깐... 머리가 안 돌아가", "하아... 생각하기 귀찮은데", "어... 대충 해도 돼?"],
        "energy_low": ["으................", "일으......... 잠깐.........", "하아.........."],
    },
    Emotion.Excited: {
        "short":      ["오!", "헐!", "잠깐잠깐!", "앗!", "엥!", "오오!"],
        "long":       ["오오오 잠깐만! 이거 진짜 좋은 생각인데!", "헐 잠깐! 갑자기 떠오른 게 있어!",
                       "앗 잠깐잠깐잠깐! 기다려봐!", "아 이거 대박인데 잠깐만!", "오! 잠깐만 정리 좀 하고!"],
        "energy_low": ["오... 잠깐만 좀..!", "앗! 근데 잠깐..."],
    },
    Emotion.Confused: {
        "short":      ["이...?", "뭔지", "잠깐?", "어?", "어...?", "이라"],
        "long":       ["이...? 잠깐만 뭐였지", "어... 아닌가? 잠깐만",
                       "어? 음 잠깐, 헷갈려", "뭔지... 잠깐만 생각 좀", "어...? 잠 아닌가. 잠깐", "이라... 잠깐 정리 좀 하고"],
        "energy_low": ["이......... 뭐였지.........", "으음...? 잠깐........."],
    },
}


class Event(str, Enum):
    Praise    = "praise"
    Scold     = "scold"
    Joke      = "joke"
    Ignore    = "ignore"
    AskHard   = "ask_hard"
    AskEasy   = "ask_easy"
    Agree     = "agree"
    Disagree  = "disagree"

EVENTS = list(Event)

EVENT_KR: dict[Event, str] = {
    Event.Praise:   "칭찬",
    Event.Scold:    "꾸중",
    Event.Joke:     "농담",
    Event.Ignore:   "무시",
    Event.AskHard:  "어려운질문",
    Event.AskEasy:  "쉬운질문",
    Event.Agree:    "동의",
    Event.Disagree: "반박",
}


# ─────────────────────────────────────────────────
#  BASE_LIKELIHOOD — 정규화 보장 (합 = 1.0)
# ─────────────────────────────────────────────────

_RAW_LIKELIHOOD: dict[Event, dict[Emotion, float]] = {
    Event.Praise:   {Emotion.Happy: 0.45, Emotion.Neutral: 0.15, Emotion.Sassy: 0.03, Emotion.Tired: 0.07, Emotion.Excited: 0.40, Emotion.Confused: 0.05},
    Event.Scold:    {Emotion.Happy: 0.03, Emotion.Neutral: 0.10, Emotion.Sassy: 0.55, Emotion.Tired: 0.12, Emotion.Excited: 0.03, Emotion.Confused: 0.10},
    Event.Joke:     {Emotion.Happy: 0.40, Emotion.Neutral: 0.12, Emotion.Sassy: 0.08, Emotion.Tired: 0.05, Emotion.Excited: 0.35, Emotion.Confused: 0.05},
    Event.Ignore:   {Emotion.Happy: 0.03, Emotion.Neutral: 0.12, Emotion.Sassy: 0.30, Emotion.Tired: 0.45, Emotion.Excited: 0.02, Emotion.Confused: 0.08},
    Event.AskHard:  {Emotion.Happy: 0.08, Emotion.Neutral: 0.18, Emotion.Sassy: 0.20, Emotion.Tired: 0.30, Emotion.Excited: 0.08, Emotion.Confused: 0.15},
    Event.AskEasy:  {Emotion.Happy: 0.30, Emotion.Neutral: 0.35, Emotion.Sassy: 0.03, Emotion.Tired: 0.08, Emotion.Excited: 0.22, Emotion.Confused: 0.03},
    Event.Agree:    {Emotion.Happy: 0.35, Emotion.Neutral: 0.25, Emotion.Sassy: 0.03, Emotion.Tired: 0.08, Emotion.Excited: 0.30, Emotion.Confused: 0.03},
    Event.Disagree: {Emotion.Happy: 0.03, Emotion.Neutral: 0.12, Emotion.Sassy: 0.45, Emotion.Tired: 0.18, Emotion.Excited: 0.05, Emotion.Confused: 0.12},
}

def _normalize(d: dict[Emotion, float]) -> dict[Emotion, float]:
    total = sum(d.values())
    return {e: v / total for e, v in d.items()}

BASE_LIKELIHOOD: dict[Event, dict[Emotion, float]] = {
    evt: _normalize(raw) for evt, raw in _RAW_LIKELIHOOD.items()
}


# ─────────────────────────────────────────────────
#  APPRAISAL_MODIFIERS — 48개 조합 (6 상태 × 8 이벤트)
# ─────────────────────────────────────────────────

APPRAISAL_MODIFIERS: dict[str, dict[Emotion, float]] = {
    # HAPPY
    "happy:praise":    {Emotion.Happy: 1.3, Emotion.Excited: 1.4},
    "happy:scold":     {Emotion.Sassy: 1.6, Emotion.Confused: 1.5, Emotion.Happy: 0.3},
    "happy:joke":      {Emotion.Happy: 1.4, Emotion.Excited: 1.3},
    "happy:ignore":    {Emotion.Sassy: 1.4, Emotion.Confused: 1.2},
    "happy:ask_hard":  {Emotion.Happy: 0.8, Emotion.Neutral: 1.3},
    "happy:ask_easy":  {Emotion.Happy: 1.2, Emotion.Excited: 1.1},
    "happy:agree":     {Emotion.Happy: 1.3, Emotion.Excited: 1.2},
    "happy:disagree":  {Emotion.Confused: 1.4, Emotion.Sassy: 1.3},
    # NEUTRAL
    "neutral:praise":   {Emotion.Happy: 1.1},
    "neutral:scold":    {Emotion.Sassy: 1.1},
    "neutral:joke":     {Emotion.Happy: 1.1},
    "neutral:ignore":   {Emotion.Tired: 1.1},
    "neutral:ask_hard": {Emotion.Neutral: 1.1},
    "neutral:ask_easy": {Emotion.Neutral: 1.1},
    "neutral:agree":    {Emotion.Happy: 1.1},
    "neutral:disagree": {Emotion.Sassy: 1.1},
    # SASSY
    "sassy:praise":    {Emotion.Happy: 1.8, Emotion.Sassy: 0.3},
    "sassy:scold":     {Emotion.Sassy: 1.6, Emotion.Tired: 1.3},
    "sassy:joke":      {Emotion.Sassy: 1.5, Emotion.Happy: 0.5},
    "sassy:ignore":    {Emotion.Sassy: 1.4, Emotion.Tired: 1.3},
    "sassy:ask_hard":  {Emotion.Sassy: 1.5, Emotion.Tired: 1.4},
    "sassy:ask_easy":  {Emotion.Sassy: 0.7, Emotion.Neutral: 1.3},
    "sassy:agree":     {Emotion.Happy: 1.4, Emotion.Sassy: 0.5},
    "sassy:disagree":  {Emotion.Sassy: 1.7, Emotion.Tired: 1.2},
    # TIRED
    "tired:praise":    {Emotion.Happy: 1.5, Emotion.Tired: 0.6},
    "tired:scold":     {Emotion.Tired: 1.3, Emotion.Sassy: 1.4},
    "tired:joke":      {Emotion.Tired: 1.2, Emotion.Happy: 0.7},
    "tired:ignore":    {Emotion.Tired: 1.5},
    "tired:ask_hard":  {Emotion.Tired: 1.8, Emotion.Sassy: 1.4},
    "tired:ask_easy":  {Emotion.Tired: 0.8, Emotion.Neutral: 1.3},
    "tired:agree":     {Emotion.Tired: 0.7, Emotion.Happy: 1.3},
    "tired:disagree":  {Emotion.Tired: 1.3, Emotion.Sassy: 1.5},
    # EXCITED
    "excited:praise":   {Emotion.Excited: 1.5, Emotion.Happy: 1.3},
    "excited:scold":    {Emotion.Confused: 1.6, Emotion.Sassy: 1.3, Emotion.Excited: 0.3},
    "excited:joke":     {Emotion.Excited: 1.4, Emotion.Happy: 1.3},
    "excited:ignore":   {Emotion.Sassy: 1.7, Emotion.Confused: 1.3, Emotion.Excited: 0.3},
    "excited:ask_hard": {Emotion.Excited: 0.6, Emotion.Neutral: 1.3},
    "excited:ask_easy": {Emotion.Excited: 1.3, Emotion.Happy: 1.2},
    "excited:agree":    {Emotion.Excited: 1.4, Emotion.Happy: 1.2},
    "excited:disagree": {Emotion.Confused: 1.5, Emotion.Sassy: 1.3},
    # CONFUSED
    "confused:praise":   {Emotion.Happy: 1.3, Emotion.Confused: 0.6},
    "confused:scold":    {Emotion.Confused: 1.4, Emotion.Sassy: 1.3},
    "confused:joke":     {Emotion.Confused: 1.3, Emotion.Happy: 0.8},
    "confused:ignore":   {Emotion.Confused: 1.3, Emotion.Tired: 1.3},
    "confused:ask_hard": {Emotion.Confused: 1.5, Emotion.Tired: 1.3},
    "confused:ask_easy": {Emotion.Confused: 0.5, Emotion.Neutral: 1.5},
    "confused:agree":    {Emotion.Happy: 1.6, Emotion.Confused: 0.4},
    "confused:disagree": {Emotion.Confused: 1.6, Emotion.Sassy: 1.2},
}


# ─────────────────────────────────────────────────
#  반대 감정 매핑 — 만족화 노이즈 감쇠용 (Enhancement 2)
# ─────────────────────────────────────────────────

OPPOSITE_EMOTIONS: dict[Emotion, Emotion] = {
    Emotion.Happy:   Emotion.Sassy,
    Emotion.Sassy:   Emotion.Happy,
    Emotion.Excited: Emotion.Tired,
    Emotion.Tired:   Emotion.Excited,
}


# ─────────────────────────────────────────────────
#  이벤트 지시 — 정적 상수 (시스템 프롬프트 base에 1회만 포함)
# ─────────────────────────────────────────────────

EMOTION_ENGINE_EVENT_INSTRUCTIONS = (
    "\n\n[감정 이벤트 지시]"
    "\n대화 중 아래 이벤트를 감지하면 반드시 해당 ACT 토큰을 응답에 포함하라."
    '\n형식: <|ACT: {"event": "<이벤트명>"}|>'
    "\n- praise: 상대가 나를 칭찬하거나 긍정적으로 평가할 때"
    "\n- scold: 상대가 나를 꾸짖거나 부정적으로 비판할 때"
    "\n- joke: 농담이나 유머러스한 상호작용이 있을 때"
    "\n- ignore: 내 말이 무시되거나 대화가 단절될 때"
    "\n- ask_hard: 어렵거나 복잡한 질문을 받을 때"
    "\n- ask_easy: 간단하고 쉬운 질문을 받을 때"
    "\n- agree: 상대가 내 의견에 동의하거나 공감할 때"
    "\n- disagree: 상대가 내 의견에 반박하거나 이의를 제기할 때"
    '\n(기존 <|ACT: {"emotion": "..."}|> 형식도 병행 사용 가능)'
)


# ─────────────────────────────────────────────────
#  결과 타입
# ─────────────────────────────────────────────────

@dataclass
class PeakEntry:
    emotion: Emotion
    intensity: float
    turn: int  # conversationTurnCount 기준

@dataclass
class EmotionEngineResult:
    state: Emotion
    probabilities: dict[Emotion, float]
    prev_state: Emotion
    directive: str
    ddm_steps: int
    ddm_delay_ms: int
    energy: float
    dissonance: float
    peak_memory: Optional[PeakEntry]
    decay_applied: bool
    filler: str


# ─────────────────────────────────────────────────
#  EmotionEngine
# ─────────────────────────────────────────────────

class EmotionEngine:
    """
    9이론 통합 감정 엔진.

    사용 예:
        engine = EmotionEngine(seed=42)
        result = engine.process_event(Event.Praise)
        print(result.state, result.filler)
        print(engine.build_emotion_prompt())
    """

    # ── 하이퍼파라미터 ──
    INERTIA              = 0.20   # [4] 마르코프 관성
    LAZINESS             = 0.10   # [3] 만족화 노이즈 스케일
    DDM_NOISE            = 0.08   # [2] DDM 확산 노이즈
    DDM_DRIFT_RATE       = 0.15   # [2] DDM 드리프트 계수
    DECAY_RATE           = 0.15   # [5] 쾌락 적응 decay 강도
    DECAY_TURN_INTERVAL  = 3      # [5] decay 발동 간격 (턴)
    ENERGY_DRAIN         = 0.06   # [8] 에너지 소모 기본값
    ENERGY_REGEN         = 0.08   # [8] 에너지 회복 기본값
    PEAK_WEIGHT          = 0.15   # [7] 피크 기억 편향 가중치
    PEAK_RECENCY_LIMIT   = 15     # [7] 피크 기억 유효 회차
    DISSONANCE_THRESHOLD = 0.55   # [9] 부조화 → confused 부스트 임계값

    # [8] 에너지 드레인/리젠 이벤트 배율
    _DRAIN_EVENTS  = {Event.AskHard: 2.0, Event.Scold: 1.5, Event.Disagree: 1.5, Event.Ignore: 1.2}
    _REGEN_EVENTS  = {Event.Praise: 2.0, Event.Agree: 1.5}
    _NEUTRAL_DRAIN = {Event.AskEasy: 0.5, Event.Joke: 0.3}

    # [4] 마르코프 전이 행렬 — 감정 간 자연스러운 '거리' 반영
    # 행(row) = 현재 감정, 열(col) = 다음 감정으로의 전이 선호도 (각 행 합 = 1.0)
    # 설계 원칙:
    #   - 감정 유사성: Happy↔Excited, Sassy↔Tired 는 전이 확률 높음
    #   - 감정 반발성: Happy→Tired, Excited→Sassy 는 전이 확률 낮음
    #   - 중립(Neutral)은 모든 감정에서 중간 수준의 탈출구 역할
    _MARKOV_TRANSITION: dict[Emotion, dict[Emotion, float]] = {
        #                  Happy  Neutral  Sassy  Tired  Excited  Confused
        Emotion.Happy:   {Emotion.Happy: 0.50, Emotion.Neutral: 0.20, Emotion.Sassy: 0.05,
                          Emotion.Tired: 0.05, Emotion.Excited: 0.15, Emotion.Confused: 0.05},
        Emotion.Neutral: {Emotion.Happy: 0.15, Emotion.Neutral: 0.50, Emotion.Sassy: 0.10,
                          Emotion.Tired: 0.10, Emotion.Excited: 0.10, Emotion.Confused: 0.05},
        Emotion.Sassy:   {Emotion.Happy: 0.05, Emotion.Neutral: 0.15, Emotion.Sassy: 0.45,
                          Emotion.Tired: 0.20, Emotion.Excited: 0.03, Emotion.Confused: 0.12},
        Emotion.Tired:   {Emotion.Happy: 0.05, Emotion.Neutral: 0.20, Emotion.Sassy: 0.25,
                          Emotion.Tired: 0.40, Emotion.Excited: 0.02, Emotion.Confused: 0.08},
        Emotion.Excited: {Emotion.Happy: 0.25, Emotion.Neutral: 0.15, Emotion.Sassy: 0.05,
                          Emotion.Tired: 0.15, Emotion.Excited: 0.35, Emotion.Confused: 0.05},
        Emotion.Confused:{Emotion.Happy: 0.05, Emotion.Neutral: 0.20, Emotion.Sassy: 0.20,
                          Emotion.Tired: 0.20, Emotion.Excited: 0.03, Emotion.Confused: 0.32},
    }

    # [9] 충돌 이벤트 페어 (frozenset → weight)
    _CONFLICT_PAIRS: list[tuple[frozenset[Event], float]] = [
        (frozenset({Event.Praise,   Event.Scold}),    0.8),
        (frozenset({Event.Agree,    Event.Disagree}),  0.7),
        (frozenset({Event.Joke,     Event.Scold}),     0.5),
        (frozenset({Event.Praise,   Event.Ignore}),    0.6),
        (frozenset({Event.Agree,    Event.Scold}),     0.5),
        (frozenset({Event.Praise,   Event.Disagree}),  0.5),
        (frozenset({Event.Joke,     Event.Ignore}),    0.4),
    ]

    def __init__(self, seed: Optional[int] = None):
        # [Enhancement 5] 스레드 안전성 — lock을 reset()보다 먼저 생성
        self._lock = threading.Lock()
        self.rng = SeededRandom(seed)
        self.reset()

    # ─────────────────────────────────────────────
    #  내부 헬퍼
    # ─────────────────────────────────────────────

    @staticmethod
    def _normalize(d: dict[Emotion, float]) -> dict[Emotion, float]:
        total = sum(d.values())
        return {e: v / total for e, v in d.items()}

    # ── [5] 쾌락 적응 ──

    def _apply_decay(self) -> bool:
        self.turns_since_strong_event += 1
        if self.turns_since_strong_event < self.DECAY_TURN_INTERVAL:
            return False

        # FIX: 카운터 리셋 — decayTurnInterval마다 1회만 실행
        self.turns_since_strong_event = 0

        max_prob = max(self.prior.values())
        # FIX: maxProb 비례 — 지배적일수록 decay 강하게 (역방향 버그 수정)
        intensity_factor = max(0.3, max_prob)
        decay_factor = min(0.6, self.DECAY_RATE * intensity_factor)
        baseline = 1.0 / len(EMOTIONS)
        self.prior = {
            e: v * (1 - decay_factor) + baseline * decay_factor
            for e, v in self.prior.items()
        }
        return True

    # ── [6] 평가 이론 ──

    def _appraise(self, event: Event) -> dict[Emotion, float]:
        lik = dict(BASE_LIKELIHOOD[event])
        key = f"{self.current_state.value}:{event.value}"
        mods = APPRAISAL_MODIFIERS.get(key, {})
        for emotion, multiplier in mods.items():
            lik[emotion] = lik.get(emotion, 0.01) * multiplier
        return self._normalize(lik)

    # ── [1] 베이즈 업데이트 ──

    @property
    def sentiment_baseline(self) -> dict[Emotion, float]:
        """최근 20턴 감정 이력의 빈도 분포 — 장기 호감도 기준선 (Enhancement 4)."""
        if not self.sentiment_history:
            n = len(EMOTIONS)
            return {e: 1.0 / n for e in EMOTIONS}
        counts: dict[Emotion, float] = {e: 0.0 for e in EMOTIONS}
        for e in self.sentiment_history:
            counts[e] += 1.0
        total = len(self.sentiment_history)
        return {e: counts[e] / total for e in EMOTIONS}

    def _bayes_update(self, likelihood: dict[Emotion, float]) -> dict[Emotion, float]:
        # [Enhancement 4] prior에 장기 호감도 10% 블렌딩 (sentiment baseline)
        baseline = self.sentiment_baseline
        blended_prior = self._normalize({e: 0.9 * self.prior[e] + 0.1 * baseline[e] for e in EMOTIONS})
        p_evidence = sum(likelihood[e] * blended_prior[e] for e in EMOTIONS)
        posterior = {e: (likelihood[e] * blended_prior[e]) / p_evidence for e in EMOTIONS}
        self.prior = dict(posterior)
        return posterior

    # ── [7] 피크엔드 기록 ──

    def _record_peak(self, posterior: dict[Emotion, float]) -> None:
        max_emotion = max(EMOTIONS, key=lambda e: posterior[e])
        intensity = posterior[max_emotion]
        if intensity > 0.5:
            # FIX: conversation_turn_count 기준 저장 (피크 recency 계산용)
            self.peak_memory.append(PeakEntry(max_emotion, intensity, self.conversation_turn_count))
            if len(self.peak_memory) > 5:
                self.peak_memory = self.peak_memory[-5:]
            self.turns_since_strong_event = 0

    # ── [7] 피크엔드 편향 ──

    def _apply_peak_memory(self, posterior: dict[Emotion, float]) -> dict[Emotion, float]:
        if not self.peak_memory:
            return posterior
        peak = max(self.peak_memory, key=lambda p: p.intensity)
        last = self.peak_memory[-1]
        biased = dict(posterior)
        biased[peak.emotion] = biased.get(peak.emotion, 0.0) + self.PEAK_WEIGHT * peak.intensity
        biased[last.emotion] = biased.get(last.emotion, 0.0) + self.PEAK_WEIGHT * 0.5 * last.intensity
        return self._normalize(biased)

    # ── [4] 마르코프 전이 행렬 ──

    def _markov_blend(self, posterior: dict[Emotion, float]) -> dict[Emotion, float]:
        # 전이 행렬에서 현재 감정의 선호 분포를 꺼내 INERTIA 비율로 블렌딩
        # 기존: 현재 감정에만 +INERTIA (1차원 관성)
        # 개선: 전이 행렬로 감정 간 거리(친밀/반발) 반영
        transition = self._MARKOV_TRANSITION[self.current_state]
        blended = {
            e: posterior[e] * (1 - self.INERTIA) + transition[e] * self.INERTIA
            for e in EMOTIONS
        }
        return self._normalize(blended)

    # ── [8] 에너지 업데이트 ──

    def _update_energy(self, event: Event) -> None:
        # [Enhancement 3 + 자아고갈 곡선] 점근적 자동 회복
        # += 0.01 * (1 - energy): 에너지가 낮을수록 회복 빠르고, 가득 찰수록 회복 느림
        # 1.0 근방에서 자연스럽게 수렴 (HP포션식 직선 회복 → 인간다운 피로 곡선)
        self.energy = min(1.0, self.energy + 0.01 * (1.0 - self.energy))

        # [Enhancement 3] 귀찮은 상태에서 칭찬/농담 받으면 1.5x 감동 보정
        tired_bonus = self.current_state == Emotion.Tired and event in {Event.Joke, Event.Praise}

        if event in self._DRAIN_EVENTS:
            self.energy = max(0.0, self.energy - self.ENERGY_DRAIN * self._DRAIN_EVENTS[event])
        elif event in self._REGEN_EVENTS:
            mult = self._REGEN_EVENTS[event] * (1.5 if tired_bonus else 1.0)
            self.energy = min(1.0, self.energy + self.ENERGY_REGEN * mult)
        elif event in self._NEUTRAL_DRAIN:
            if tired_bonus:
                # Joke while Tired → drain 대신 1.5x 회복으로 전환
                self.energy = min(1.0, self.energy + self.ENERGY_REGEN * 1.5)
            else:
                self.energy = max(0.0, self.energy - self.ENERGY_DRAIN * self._NEUTRAL_DRAIN[event])

    # ── [3+8] 만족화 ──

    def _satisfice(self, probs: dict[Emotion, float]) -> dict[Emotion, float]:
        energy_multiplier = 1.0 + 2.0 * (1.0 - self.energy)
        noise_scale = self.LAZINESS * energy_multiplier * (0.8 + self.rng.random() * 0.4)
        noisy: dict[Emotion, float] = {}
        for e in EMOTIONS:
            raw_noise = (self.rng.random() - 0.5) * noise_scale
            # [Enhancement 2] 노이즈 캡: 원본 값의 50% 초과 불가 (역전 방지)
            capped_noise = max(-probs[e] * 0.5, min(probs[e] * 0.5, raw_noise))
            # [Enhancement 2] 현재 감정의 반대 감정은 노이즈를 절반으로 감쇠 (Happy↔Sassy, Excited↔Tired)
            if OPPOSITE_EMOTIONS.get(self.current_state) == e:
                capped_noise *= 0.5
            noisy[e] = max(0.01, probs[e] + capped_noise)
        if self.energy < 0.3:
            noisy[Emotion.Tired] *= 1.5
            noisy[Emotion.Sassy] *= 1.3
        return self._normalize(noisy)

    # ── [9] 인지 부조화 ──

    def _compute_dissonance(self, event: Event) -> float:
        self.recent_events.append(event)
        if len(self.recent_events) > 5:
            self.recent_events = self.recent_events[-5:]
        if len(self.recent_events) < 2:
            self.dissonance = 0.0
            return 0.0

        # FIX: 윈도우 4→5 통일 (슬라이스 제거, 전체 사용)
        recent = self.recent_events
        total_conflict = 0.0
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                pair = frozenset({recent[i], recent[j]})
                for conflict_pair, weight in self._CONFLICT_PAIRS:
                    if pair == conflict_pair:
                        recency = 1.0 - (j - i) * 0.2
                        total_conflict += weight * recency
                        break

        self.dissonance = min(1.0, self.dissonance * 0.6 + total_conflict)
        return self.dissonance

    def _apply_dissonance_to_probs(self, probs: dict[Emotion, float]) -> dict[Emotion, float]:
        if self.dissonance <= self.DISSONANCE_THRESHOLD:
            return probs
        boost = self.dissonance * 0.25
        adjusted = dict(probs)
        adjusted[Emotion.Confused] = adjusted.get(Emotion.Confused, 0.0) + boost
        return self._normalize(adjusted)

    # ── [2] DDM ──

    def _ddm_decide(self, probs: dict[Emotion, float]) -> tuple[Emotion, int, int]:
        # [Enhancement 1] 엔트로피 기반 임계값 조정 — 분포가 불확실할수록 빠르게 결정
        n = len(EMOTIONS)
        entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())
        entropy_ratio = entropy / math.log(n)  # 0(확실) ~ 1(균등)

        base_threshold = 0.3 + self.rng.random() * 0.2
        # 엔트로피↑ → 임계값↓ → 더 빨리 결정 (불확실할 때 무한 지연 방지)
        base_threshold -= entropy_ratio * 0.15
        dissonance_wobble = self.dissonance * (self.rng.random() - 0.5) * 0.1
        threshold = max(0.15, base_threshold + dissonance_wobble)

        accum = {e: 0.0 for e in EMOTIONS}
        steps = 0
        winner: Optional[Emotion] = None

        while winner is None and steps < 60:
            steps += 1
            for e in EMOTIONS:
                drift = probs[e] * self.DDM_DRIFT_RATE
                noise = (self.rng.random() - 0.5) * self.DDM_NOISE
                accum[e] += drift + noise

            # FIX: step 완료 후 crossed 감지 → argmax로 winner 결정 (순서 편향 제거)
            crossed = [e for e in EMOTIONS if accum[e] >= threshold]
            if crossed:
                winner = max(crossed, key=lambda e: accum[e])

        if winner is None:
            winner = max(EMOTIONS, key=lambda e: accum[e])

        # [Enhancement 1] 확률이 높을수록 delay 단축 (결단력) + 최대 1200ms 캡
        max_prob = max(probs.values())
        raw_delay = int(300 + steps * 70 + self.rng.random() * 200 + self.dissonance * 300)
        raw_delay -= int(max_prob * 200)
        delay_ms = min(max(0, raw_delay), 1200)
        return winner, steps, delay_ms

    # ── 추임새 선택 ──

    def _pick_filler(self, state: Emotion, delay_ms: int) -> str:
        fillers = EMOTION_FILLERS[state]
        if self.energy < 0.3 and fillers["energy_low"]:
            pool = fillers["energy_low"]
        elif delay_ms > 800:
            pool = fillers["long"]
        else:
            pool = fillers["short"]
        return pool[int(self.rng.random() * len(pool))]

    # ─────────────────────────────────────────────
    #  퍼블릭 API
    # ─────────────────────────────────────────────

    def process_event(self, event: Event | str) -> EmotionEngineResult:
        """
        엔진 이벤트를 처리해 감정 상태를 업데이트한다.
        알 수 없는 이벤트 문자열은 경고 후 no-op 반환 (NaN 전파 방지).
        """
        with self._lock:  # [Enhancement 5]
            # FIX: unknown event 가드 — NaN 전파 방지
            if isinstance(event, str):
                try:
                    event = Event(event)
                except ValueError:
                    warnings.warn(f"[EmotionEngine] Unknown event '{event}', skipping.")
                    return EmotionEngineResult(
                        state=self.current_state,
                        probabilities=dict(self.prior),
                        prev_state=self.current_state,
                        directive=EMOTION_DIRECTIVES[self.current_state],
                        ddm_steps=0, ddm_delay_ms=0,
                        energy=self.energy, dissonance=self.dissonance,
                        peak_memory=self.peak_memory[-1] if self.peak_memory else None,
                        decay_applied=False, filler="",
                    )

            self.turn_count += 1
            # FIX: conversationTurnCount — processEvent 경로에서도 증가
            self.conversation_turn_count += 1
            prev = self.current_state

            decay_applied  = self._apply_decay()      # [5]
            self._update_energy(event)                # [8]
            self._compute_dissonance(event)           # [9]
            likelihood     = self._appraise(event)    # [6]
            posterior      = self._bayes_update(likelihood)  # [1]
            self._record_peak(posterior)              # [7] 기록
            posterior      = self._apply_peak_memory(posterior)  # [7] 편향
            blended        = self._markov_blend(posterior)    # [4]
            blended        = self._satisfice(blended)         # [3+8]
            blended        = self._apply_dissonance_to_probs(blended)  # [9]
            winner, steps, delay_ms = self._ddm_decide(blended)  # [2]

            self.current_state = winner
            filler = self._pick_filler(winner, delay_ms)

            # [Enhancement 4] 장기 호감도 이력 업데이트 (최근 20턴)
            self.sentiment_history.append(winner)
            if len(self.sentiment_history) > 20:
                self.sentiment_history = self.sentiment_history[-20:]

            return EmotionEngineResult(
                state=winner,
                probabilities=blended,
                prev_state=prev,
                directive=EMOTION_DIRECTIVES[winner],
                ddm_steps=steps,
                ddm_delay_ms=delay_ms,
                energy=self.energy,
                dissonance=self.dissonance,
                peak_memory=self.peak_memory[-1] if self.peak_memory else None,
                decay_applied=decay_applied,
                filler=filler,
            )

    def set_emotion_direct(self, emotion: Emotion | str) -> EmotionEngineResult:
        """
        패스스루 모드 — 기존 emotion ACT 하위 호환.
        엔진 파이프라인을 우회하고 상태만 직접 설정.
        에너지/부조화는 건드리지 않음 (연속성 유지).
        """
        with self._lock:  # [Enhancement 5]
            if isinstance(emotion, str):
                emotion = Emotion(emotion)

            prev = self.current_state
            self.current_state = emotion
            # FIX: conversationTurnCount — set_emotion_direct 경로에서도 증가
            self.conversation_turn_count += 1

            # prior를 해당 감정 중심으로 soft-set
            n = len(EMOTIONS)
            self.prior = {
                e: 0.6 if e == emotion else 0.4 / (n - 1)
                for e in EMOTIONS
            }

            # [Enhancement 4] 장기 호감도 이력 업데이트 (최근 20턴)
            self.sentiment_history.append(emotion)
            if len(self.sentiment_history) > 20:
                self.sentiment_history = self.sentiment_history[-20:]

            return EmotionEngineResult(
                state=emotion,
                probabilities=dict(self.prior),
                prev_state=prev,
                directive=EMOTION_DIRECTIVES[emotion],
                ddm_steps=0, ddm_delay_ms=0,
                energy=self.energy, dissonance=self.dissonance,
                peak_memory=self.peak_memory[-1] if self.peak_memory else None,
                decay_applied=False, filler="",
            )

    def build_emotion_prompt(self) -> str:
        """
        현재 감정 상태를 LLM 시스템 프롬프트에 삽입할 동적 블록을 반환한다.
        정적인 이벤트 지시(EMOTION_ENGINE_EVENT_INSTRUCTIONS)는 별도로 base에 추가할 것.
        """
        with self._lock:  # [Enhancement 5]
            energy_desc = ""
            if self.energy < 0.3:
                energy_desc = "\n감정 에너지가 바닥나 짜증이 나기 쉬운 상태다."
            elif self.energy < 0.6:
                energy_desc = "\n감정 에너지가 조금 부족해 평소보다 덜 참을성이 있다."

            dissonance_desc = ""
            if self.dissonance > 0.5:
                dissonance_desc = "\n최근 상반된 반응에 감정적으로 혼란스럽다. 확신 없이 말하는 경향이 있다."

            # FIX: 15턴 이내의 피크만 표시 — 오래된 기억은 자연스럽게 소멸
            peak_desc = ""
            if self.peak_memory:
                peak = max(self.peak_memory, key=lambda p: p.intensity)
                turns_ago = self.conversation_turn_count - peak.turn
                if turns_ago <= self.PEAK_RECENCY_LIMIT:
                    peak_desc = f"\n과거 대화에서 강하게 '{EMOTION_KR[peak.emotion]}' 감정을 느꼈던 기억이 남아있다."

            return (
                f"\n\n[감정 상태: {EMOTION_KR[self.current_state]}]"
                f"\n{EMOTION_DIRECTIVES[self.current_state]}"
                + energy_desc
                + dissonance_desc
                + peak_desc
                + f"\n(대화 회차: {self.conversation_turn_count}회차 | 에너지: {round(self.energy * 100)}%)"
            )

    # ── 직렬화 ──

    def serialize(self) -> dict:
        with self._lock:  # [Enhancement 5]
            return {
                "current_state":           self.current_state.value,
                "turn_count":              self.turn_count,
                "conversation_turn_count": self.conversation_turn_count,
                "energy":                  self.energy,
                "dissonance":              self.dissonance,
                "prior":                   {e.value: v for e, v in self.prior.items()},
                "peak_memory":             [dataclasses.asdict(p) for p in self.peak_memory],
                "recent_events":           [e.value for e in self.recent_events],
                "turns_since_strong_event": self.turns_since_strong_event,
                "rng_state":               list(self.rng.get_state()),
                # [Enhancement 4]
                "sentiment_history":       [e.value for e in self.sentiment_history],
            }

    def restore(self, data: dict) -> None:
        with self._lock:  # [Enhancement 5]
            self.current_state           = Emotion(data["current_state"])
            self.turn_count              = data["turn_count"]
            # FIX: 구버전 직렬화 데이터 호환 fallback
            self.conversation_turn_count = data.get("conversation_turn_count", data["turn_count"])
            self.energy                  = data["energy"]
            self.dissonance              = data["dissonance"]
            self.prior                   = {Emotion(k): v for k, v in data["prior"].items()}
            self.peak_memory             = [PeakEntry(Emotion(p["emotion"]), p["intensity"], p["turn"]) for p in data["peak_memory"]]
            self.recent_events           = [Event(e) for e in data["recent_events"]]
            self.turns_since_strong_event = data["turns_since_strong_event"]
            self.rng.set_state(tuple(data["rng_state"]))  # type: ignore[arg-type]
            # [Enhancement 4] 구버전 호환 fallback
            self.sentiment_history       = [Emotion(e) for e in data.get("sentiment_history", [])]

    def reset(self) -> None:
        with self._lock:  # [Enhancement 5]
            n = len(EMOTIONS)
            self.current_state: Emotion = Emotion.Neutral
            self.turn_count: int = 0
            self.conversation_turn_count: int = 0
            self.energy: float = 1.0
            self.prior: dict[Emotion, float] = {e: 1.0 / n for e in EMOTIONS}
            self.peak_memory: list[PeakEntry] = []
            self.recent_events: list[Event] = []
            self.dissonance: float = 0.0
            self.turns_since_strong_event: int = 0
            self.sentiment_history: list[Emotion] = []


# ─────────────────────────────────────────────────
#  빠른 시뮬레이션 (CLI 확인용)
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    engine = EmotionEngine(seed=42)

    scenarios = [
        ("칭찬 → 칭찬 → 칭찬",        [Event.Praise,   Event.Praise,   Event.Praise]),
        ("꾸중 × 5",                   [Event.Scold] * 5),
        ("칭찬 ↔ 꾸중 교대 (부조화)", [Event.Praise, Event.Scold, Event.Praise, Event.Scold]),
        ("어려운질문 × 10 (에너지 고갈)", [Event.AskHard] * 10),
        ("칭찬으로 회복",              [Event.Praise, Event.Praise, Event.Agree]),
    ]

    for title, events in scenarios:
        engine.reset()
        print(f"\n{'='*60}")
        print(f"시나리오: {title}")
        print(f"{'='*60}")
        for evt in events:
            r = engine.process_event(evt)
            prob_str = "  ".join(
                f"{EMOTION_KR[e]}={v:.2f}"
                for e, v in sorted(r.probabilities.items(), key=lambda x: -x[1])
            )
            print(
                f"  [{EVENT_KR[evt]:6s}] → {EMOTION_KR[r.state]:5s}"
                f"  에너지={r.energy:.2f}  부조화={r.dissonance:.2f}"
                f"  steps={r.ddm_steps:2d}  filler='{r.filler}'"
            )
            print(f"          분포: {prob_str}")

        print("\n[시스템 프롬프트 (동적 부분)]")
        print(engine.build_emotion_prompt())

    # 직렬화 라운드트립 테스트
    engine.reset()
    engine.process_event(Event.Praise)
    engine.process_event(Event.Joke)
    saved = engine.serialize()
    engine2 = EmotionEngine(seed=0)
    engine2.restore(saved)
    assert engine2.current_state == engine.current_state
    assert abs(engine2.energy - engine.energy) < 1e-9
    assert engine2.conversation_turn_count == engine.conversation_turn_count
    print("\n[OK] 직렬화/복원 라운드트립 통과")
