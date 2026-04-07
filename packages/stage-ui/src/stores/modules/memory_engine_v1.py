"""
Memory Engine v1 — 차별 기억 엔진 (감정-기억 통합 인지 아키텍처)
================================================================
emotion_engine_v3.py 와 연동하여 사용하는 독립 실행 가능 기억 엔진.

[이론 기반]
1. ACT-R 기저 활성화 (Anderson, 1983)         — 멱법칙 망각 + 간격 반복 내재화
2. eCMR 코사인 유사도 (Polyn et al., 2009)    — 연속형 감정 벡터 기반 기분 일치 인출
3. Yerkes-Dodson 법칙 (1908)                   — 에너지/부조화 조합별 인코딩 품질 분기
4. Flashbulb Memory (Brown & Kulik, 1977)      — 고각성+풍부한 에너지 순간 망각 면역
5. 주기적 GC 수면 사이클                       — 비활성 기억 제거 (LLM 호출 없음)

[과잉 설계 항목 간소화]
- A-MAC 5차원 → importance = max(emotion_dist) * (1 + dissonance * 0.5)
- ACAN 크로스 어텐션 → eCMR 코사인 유사도로 흡수
- 수면 사이클 LLM 병합 → GC만 (v2에서 의미 기억 병합 추가)

[감정 엔진 v3 연동 포인트]
- EmotionEngineResult.probabilities → emotion_dist (6-dim vector)
- EmotionEngineResult.energy        → 인코딩 품질 게이트
- EmotionEngineResult.dissonance    → importance 부스트 + flashbulb 트리거
- EmotionEngineResult.state         → dominant_emotion 라벨
"""

from __future__ import annotations

import dataclasses
import math
import threading
import uuid
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────
#  감정 정의 (emotion_engine_v3 와 동기화)
# ─────────────────────────────────────────────────

class Emotion(str, Enum):
    Happy    = "happy"
    Neutral  = "neutral"
    Sassy    = "sassy"
    Tired    = "tired"
    Excited  = "excited"
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


# ─────────────────────────────────────────────────
#  기억 타입
# ─────────────────────────────────────────────────

class MemoryType(str, Enum):
    Episodic = "episodic"  # 단기적 경험/일화
    Semantic = "semantic"  # 압축된 장기 지식 (v2 수면 사이클에서 생성)


# ─────────────────────────────────────────────────
#  인코딩 품질 — Yerkes-Dodson 법칙 기반
# ─────────────────────────────────────────────────

class EncodingQuality(str, Enum):
    Full       = "full"        # 완전 인코딩 (LTP 최적 조건)
    Fragmented = "fragmented"  # 파편화 인코딩 (고부조화 + 에너지 고갈)
    Degraded   = "degraded"    # 저품질 인코딩 (일상적 소진)


# ─────────────────────────────────────────────────
#  기억 단위 (MemoryEntry)
# ─────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    memory_id:        str                  # UUID — 고유 식별자
    content:          str                  # 기억 내용 (파편화 시 잘릴 수 있음)
    memory_type:      MemoryType           # Episodic / Semantic
    emotion_dist:     list[float]          # 저장 당시 6-dim 감정 확률 분포 (eCMR용)
    dominant_emotion: Emotion              # DDM 승자 감정 라벨
    energy:           float                # 저장 당시 에너지 (인코딩 품질 결정)
    dissonance:       float                # 저장 당시 부조화 (중요도 부스트)
    importance:       float                # 0.0 ~ 1.0 (A-MAC 간소화 스코어)
    created_turn:     int                  # 생성 회차 (conversationTurnCount 기준)
    access_history:   list[int]            # 인출 시점 이력 — ACT-R Σ 연산용
    flashbulb:        bool = False         # 망각 면역 플래그
    encoding_quality: EncodingQuality = EncodingQuality.Full


# ─────────────────────────────────────────────────
#  인출 결과
# ─────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    memory:     MemoryEntry
    score:      float   # 최종 인출 스코어 (B_i + eCMR 가중치)
    activation: float   # ACT-R 기저 활성화 B_i
    mood_match: float   # eCMR 코사인 유사도 기반 감정 일치도


# ─────────────────────────────────────────────────
#  MemoryEngine
# ─────────────────────────────────────────────────

class MemoryEngine:
    """
    차별 기억 엔진 v1.

    사용 예 (emotion_engine_v3 연동):
        mem = MemoryEngine()
        result = emotion_engine.process_event(Event.Praise)
        mem.tick(emotion_engine.conversation_turn_count)
        mem.store(
            content          = "형님이 칭찬해줬다",
            emotion_dist     = list(result.probabilities.values()),
            dominant_emotion = result.state,
            energy           = result.energy,
            dissonance       = result.dissonance,
        )
        prompt = mem.build_memory_prompt(list(result.probabilities.values()))
    """

    # ── 하이퍼파라미터 ──
    DECAY_BASE            = 0.5    # ACT-R 기본 감쇠 계수 d
    DECAY_FLASHBULB       = 0.05   # flashbulb/semantic — 사실상 영구 보존
    ECMR_GAMMA            = 0.5    # eCMR 감정 일치도 스케일 γ
    ACTIVATION_THRESHOLD  = -3.0   # GC 임계값 (B_i 이하 기억 삭제)
    SLEEP_CYCLE_INTERVAL  = 20     # 수면 사이클 발동 간격 (턴)
    MAX_MEMORIES          = 200    # 최대 기억 보유 수

    # Yerkes-Dodson 임계값
    YD_HIGH_DISSONANCE    = 0.8
    YD_HIGH_ENERGY        = 0.5
    YD_LOW_ENERGY         = 0.2
    YD_LOW_DISSONANCE     = 0.3
    YD_EXHAUSTED_ENERGY   = 0.1

    def __init__(self):
        self._lock = threading.Lock()
        self._init_state()

    def _init_state(self) -> None:
        """락 없이 상태 초기화 — __init__ 및 reset() 내부에서 사용."""
        self.memories: list[MemoryEntry] = []
        self.current_turn: int = 0
        self._turns_since_sleep: int = 0

    # ─────────────────────────────────────────────
    #  내부 헬퍼
    # ─────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """코사인 유사도 — eCMR 감정 벡터 간 유사도 측정."""
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)

    def _decay_param(self, m: MemoryEntry) -> float:
        """
        ACT-R 감쇠 계수 d_i.
        - flashbulb / Semantic → 0.05 (영구 보존에 가까움)
        - 일반 기억            → 0.5 * (1 - importance) (중요할수록 느리게 감쇠)
        """
        if m.flashbulb or m.memory_type == MemoryType.Semantic:
            return self.DECAY_FLASHBULB
        return max(0.05, self.DECAY_BASE * (1.0 - m.importance))

    def _base_activation(self, m: MemoryEntry) -> float:
        """
        ACT-R 기저 활성화 수준 B_i.

        B_i = ln( Σ_{t_k ∈ access_history} (current_turn - t_k + 1)^(-d) )

        - 멱법칙 망각: 지수 감쇠보다 오래된 기억이 더 잘 살아남음
        - 간격 반복 내재화: 자주 인출될수록 Σ 항이 누적 → B_i↑
        """
        d = self._decay_param(m)
        history = m.access_history if m.access_history else [m.created_turn]
        total = sum(
            (self.current_turn - t + 1) ** (-d)
            for t in history
            if self.current_turn - t + 1 > 0
        )
        if total <= 0:
            return float('-inf')
        return math.log(total)

    def _retrieval_score(
        self,
        m: MemoryEntry,
        current_dist: list[float],
    ) -> tuple[float, float, float]:
        """
        최종 인출 스코어 = B_i + γ * cosine_similarity(enc_dist, cur_dist)

        Returns: (score, activation, mood_match)
        """
        activation  = self._base_activation(m)
        cosine      = self._cosine_similarity(m.emotion_dist, current_dist)
        mood_match  = 1.0 + self.ECMR_GAMMA * cosine
        score       = activation + self.ECMR_GAMMA * cosine
        return score, activation, mood_match

    @staticmethod
    def _encoding_quality(energy: float, dissonance: float) -> EncodingQuality:
        """
        Yerkes-Dodson 역U자 법칙 기반 인코딩 품질.

        고부조화 + 충분한 에너지  → Full       (편도체 LTP 최적)
        고부조화 + 에너지 고갈    → Fragmented (인지 과부하, 파편화)
        일상적 소진               → Degraded   (저해상도, 빠른 망각)
        그 외                     → Full
        """
        if dissonance > 0.8 and energy > 0.5:
            return EncodingQuality.Full
        if dissonance > 0.8 and energy < 0.2:
            return EncodingQuality.Fragmented
        if dissonance < 0.3 and energy < 0.1:
            return EncodingQuality.Degraded
        return EncodingQuality.Full

    @staticmethod
    def _is_flashbulb(energy: float, dissonance: float) -> bool:
        """
        Flashbulb 조건 (Yerkes-Dodson 역U자 정점):
        높은 각성(dissonance↑) + 충분한 인지 자원(energy↑) → 망각 면역
        NOTICE: energy < 0.1 → flashbulb 는 Yerkes-Dodson 위반이므로 사용하지 않음.
                에너지 고갈 상태는 오히려 Fragmented 인코딩으로 분기.
        """
        return dissonance > 0.8 and energy > 0.5

    @staticmethod
    def _calc_importance(emotion_dist: list[float], dissonance: float) -> float:
        """
        A-MAC 간소화 중요도.
        importance = max(emotion_dist) * (1 + dissonance * 0.5)
        LLM 호출 없이 감정 엔진 상태만으로 즉시 산출.
        """
        peak = max(emotion_dist) if emotion_dist else 0.0
        return min(1.0, peak * (1.0 + 0.5 * dissonance))

    @staticmethod
    def _fragment_content(content: str) -> str:
        """
        파편화 인코딩: 앞 절반 단어만 보존 + 맥락 손실 표시.
        에너지 고갈 + 고부조화 → 불완전한 기억 형성 시뮬레이션.
        """
        words = content.split()
        kept  = words[:max(3, len(words) // 2)]
        return " ".join(kept) + "... (맥락 손실)"

    def _evict_one(self) -> None:
        """MAX_MEMORIES 초과 시 B_i 가장 낮은 기억 1개 제거 (flashbulb 제외)."""
        candidates = [m for m in self.memories if not m.flashbulb]
        if not candidates:
            return
        worst = min(candidates, key=lambda m: self._base_activation(m))
        self.memories.remove(worst)

    # ─────────────────────────────────────────────
    #  퍼블릭 API
    # ─────────────────────────────────────────────

    def tick(self, turn: Optional[int] = None) -> None:
        """
        턴 증가 + 주기적 수면 사이클 트리거.
        EmotionEngine의 conversation_turn_count 와 동기화해서 호출.
        """
        with self._lock:
            if turn is not None:
                self.current_turn = turn
            else:
                self.current_turn += 1
            self._turns_since_sleep += 1
            if self._turns_since_sleep >= self.SLEEP_CYCLE_INTERVAL:
                self._sleep_cycle_unlocked()

    def store(
        self,
        content:          str,
        emotion_dist:     list[float],
        dominant_emotion: Emotion | str,
        energy:           float,
        dissonance:       float,
        memory_type:      MemoryType = MemoryType.Episodic,
    ) -> MemoryEntry:
        """
        기억을 저장한다.

        Args:
            content:          기억할 텍스트 (대화 내용, 사건 요약 등)
            emotion_dist:     EmotionEngineResult.probabilities.values() 리스트
            dominant_emotion: EmotionEngineResult.state
            energy:           EmotionEngineResult.energy
            dissonance:       EmotionEngineResult.dissonance
        """
        with self._lock:
            if isinstance(dominant_emotion, str):
                dominant_emotion = Emotion(dominant_emotion)

            quality   = self._encoding_quality(energy, dissonance)
            flashbulb = self._is_flashbulb(energy, dissonance)

            # 파편화 인코딩: 내용 손상 시뮬레이션
            stored_content = (
                self._fragment_content(content)
                if quality == EncodingQuality.Fragmented
                else content
            )

            importance = self._calc_importance(emotion_dist, dissonance)
            # Degraded 인코딩 → 중요도 강제 하향 (빠른 망각 유도)
            if quality == EncodingQuality.Degraded:
                importance *= 0.4

            entry = MemoryEntry(
                memory_id        = str(uuid.uuid4()),
                content          = stored_content,
                memory_type      = memory_type,
                emotion_dist     = list(emotion_dist),
                dominant_emotion = dominant_emotion,
                energy           = energy,
                dissonance       = dissonance,
                importance       = importance,
                created_turn     = self.current_turn,
                access_history   = [self.current_turn],
                flashbulb        = flashbulb,
                encoding_quality = quality,
            )

            self.memories.append(entry)

            # 최대 기억 수 초과 → 최저 활성화 기억 제거 (flashbulb 제외)
            if len(self.memories) > self.MAX_MEMORIES:
                self._evict_one()

            return entry

    def retrieve(
        self,
        emotion_dist: list[float],
        top_k:        int = 3,
    ) -> list[RetrievalResult]:
        """
        현재 감정 벡터 기준으로 가장 관련성 높은 기억 top_k개 반환.
        인출 시 access_history에 현재 턴 추가 → 간격 반복 효과 자동 반영.
        """
        with self._lock:
            if not self.memories:
                return []

            scored: list[RetrievalResult] = []
            for m in self.memories:
                score, activation, mood = self._retrieval_score(m, emotion_dist)
                scored.append(RetrievalResult(m, score, activation, mood))

            scored.sort(key=lambda r: r.score, reverse=True)
            top = scored[:top_k]

            # 인출 이력 갱신 (다음 B_i 계산에 반영)
            for r in top:
                r.memory.access_history.append(self.current_turn)

            return top

    def build_memory_prompt(
        self,
        emotion_dist: list[float],
        top_k:        int = 3,
    ) -> str:
        """
        인출된 기억을 LLM 시스템 프롬프트 삽입용 텍스트로 변환.
        EmotionEngine.build_emotion_prompt() 뒤에 이어 붙이면 된다.
        """
        results = self.retrieve(emotion_dist, top_k=top_k)
        if not results:
            return ""

        lines = ["\n\n[기억]"]
        for r in results:
            m = r.memory
            flag = " ★" if m.flashbulb else ""
            quality_tag = {
                EncodingQuality.Fragmented: " [불완전]",
                EncodingQuality.Degraded:   " [희미]",
            }.get(m.encoding_quality, "")
            emotion_label = EMOTION_KR.get(m.dominant_emotion, m.dominant_emotion.value)
            lines.append(
                f"- ({emotion_label}{flag}{quality_tag}) {m.content}"
            )
        return "\n".join(lines)

    # ── 수면 사이클 (GC) ──

    def _sleep_cycle_unlocked(self) -> int:
        """
        내부 GC — 락 보유 상태에서 호출.
        B_i < ACTIVATION_THRESHOLD 이고 flashbulb/semantic 이 아닌 기억 제거.
        """
        before = len(self.memories)
        self.memories = [
            m for m in self.memories
            if m.flashbulb
            or m.memory_type == MemoryType.Semantic
            or self._base_activation(m) > self.ACTIVATION_THRESHOLD
        ]
        self._turns_since_sleep = 0
        removed = before - len(self.memories)
        if removed > 0:
            pass  # NOTICE: 로깅 필요 시 여기에 추가
        return removed

    def sleep_cycle(self) -> int:
        """외부에서 수동으로 수면 사이클 트리거. 제거된 기억 수 반환."""
        with self._lock:
            return self._sleep_cycle_unlocked()

    # ── 직렬화 ──

    def serialize(self) -> dict:
        with self._lock:
            return {
                "current_turn":      self.current_turn,
                "turns_since_sleep": self._turns_since_sleep,
                "memories": [
                    {
                        **dataclasses.asdict(m),
                        "memory_type":      m.memory_type.value,
                        "dominant_emotion": m.dominant_emotion.value,
                        "encoding_quality": m.encoding_quality.value,
                    }
                    for m in self.memories
                ],
            }

    def restore(self, data: dict) -> None:
        with self._lock:
            self.current_turn       = data["current_turn"]
            self._turns_since_sleep = data.get("turns_since_sleep", 0)
            self.memories = [
                MemoryEntry(
                    memory_id        = m["memory_id"],
                    content          = m["content"],
                    memory_type      = MemoryType(m["memory_type"]),
                    emotion_dist     = m["emotion_dist"],
                    dominant_emotion = Emotion(m["dominant_emotion"]),
                    energy           = m["energy"],
                    dissonance       = m["dissonance"],
                    importance       = m["importance"],
                    created_turn     = m["created_turn"],
                    access_history   = m["access_history"],
                    flashbulb        = m["flashbulb"],
                    encoding_quality = EncodingQuality(m["encoding_quality"]),
                )
                for m in data["memories"]
            ]

    def reset(self) -> None:
        with self._lock:
            self._init_state()


# ─────────────────────────────────────────────────
#  빠른 시뮬레이션 (CLI 확인용)
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    mem = MemoryEngine()

    # ── 균등 분포 헬퍼 ──
    def uniform_dist() -> list[float]:
        n = len(EMOTIONS)
        return [1.0 / n] * n

    def peaked_dist(emotion: Emotion, strength: float = 0.7) -> list[float]:
        n = len(EMOTIONS)
        rest = (1.0 - strength) / (n - 1)
        return [strength if e == emotion else rest for e in EMOTIONS]

    print("=" * 60)
    print("시나리오 1: 기억 저장 및 감정 기반 인출")
    print("=" * 60)

    # 칭찬받아 기분 좋을 때 저장
    mem.tick(1)
    mem.store("형님이 오늘 코드 리뷰에서 칭찬해줬다.",
              peaked_dist(Emotion.Happy, 0.7), Emotion.Happy, energy=0.9, dissonance=0.1)

    # 꾸중받아 깐찐할 때 저장
    mem.tick(2)
    mem.store("버그 때문에 혼났다. 기분이 나쁘다.",
              peaked_dist(Emotion.Sassy, 0.6), Emotion.Sassy, energy=0.7, dissonance=0.3)

    # 칭찬+부조화 → flashbulb 조건
    mem.tick(3)
    entry = mem.store("완전히 예상 못한 칭찬을 받았다. 충격적일 만큼 기뻤다.",
                      peaked_dist(Emotion.Excited, 0.75), Emotion.Excited,
                      energy=0.8, dissonance=0.85)
    print(f"  Flashbulb 생성: {entry.flashbulb} / 품질: {entry.encoding_quality.value}")

    # 에너지 고갈 + 고부조화 → Fragmented 인코딩
    mem.tick(4)
    entry2 = mem.store("정말 힘든 상황인데 뭔가 중요한 일이 있었는데 기억이 안 난다.",
                       peaked_dist(Emotion.Confused, 0.5), Emotion.Confused,
                       energy=0.15, dissonance=0.9)
    print(f"  Fragmented 생성: {entry2.encoding_quality.value}")
    print(f"  저장된 내용: '{entry2.content}'")

    # 일상 소진 → Degraded
    mem.tick(5)
    entry3 = mem.store("그냥 오늘 뭔가 했던 것 같다.",
                       peaked_dist(Emotion.Tired, 0.4), Emotion.Tired,
                       energy=0.08, dissonance=0.1)
    print(f"  Degraded 생성: {entry3.encoding_quality.value} / importance={entry3.importance:.3f}")

    print(f"\n총 기억 수: {len(mem.memories)}")

    print("\n" + "=" * 60)
    print("시나리오 2: Happy 상태에서 인출 → Happy 기억 우선")
    print("=" * 60)
    mem.tick(6)
    results = mem.retrieve(peaked_dist(Emotion.Happy, 0.75), top_k=3)
    for r in results:
        print(f"  [{r.memory.dominant_emotion.value:8s}] score={r.score:+.3f}"
              f"  B_i={r.activation:+.3f}  mood={r.mood_match:.3f}"
              f"  flash={r.memory.flashbulb}"
              f"  | {r.memory.content[:40]}")

    print("\n" + "=" * 60)
    print("시나리오 3: Sassy 상태에서 인출 → 부정 기억 우선")
    print("=" * 60)
    mem.tick(7)
    results = mem.retrieve(peaked_dist(Emotion.Sassy, 0.75), top_k=3)
    for r in results:
        print(f"  [{r.memory.dominant_emotion.value:8s}] score={r.score:+.3f}"
              f"  B_i={r.activation:+.3f}  mood={r.mood_match:.3f}"
              f"  | {r.memory.content[:40]}")

    print("\n" + "=" * 60)
    print("시나리오 4: 간격 반복 - 자주 인출된 기억의 B_i 상승")
    print("=" * 60)
    happy_dist = peaked_dist(Emotion.Happy, 0.75)
    for turn in range(8, 14):
        mem.tick(turn)
        mem.retrieve(happy_dist, top_k=1)  # Happy 기억 반복 인출

    # B_i 변화 확인
    for m in mem.memories:
        b = mem._base_activation(m)
        print(f"  [{m.dominant_emotion.value:8s}] B_i={b:+.3f}"
              f"  access_count={len(m.access_history)}"
              f"  flash={m.flashbulb}")

    print("\n" + "=" * 60)
    print("시나리오 5: 시스템 프롬프트 빌드")
    print("=" * 60)
    mem.tick(14)
    prompt = mem.build_memory_prompt(peaked_dist(Emotion.Happy, 0.7))
    print(prompt)

    print("\n" + "=" * 60)
    print("시나리오 6: 수면 사이클 GC")
    print("=" * 60)
    # 20턴 이상 경과 → GC 자동 발동
    for t in range(15, 40):
        mem.tick(t)
    before = len(mem.memories)
    # 이미 tick에서 자동 GC 돌았을 것
    after = len(mem.memories)
    print(f"  GC 후 기억 수: {before} → {after}")
    # flashbulb 기억은 살아있어야 함
    flashbulbs = [m for m in mem.memories if m.flashbulb]
    print(f"  Flashbulb 보존: {len(flashbulbs)}개")

    print("\n" + "=" * 60)
    print("시나리오 7: 직렬화/복원 라운드트립")
    print("=" * 60)
    saved = mem.serialize()
    mem2 = MemoryEngine()
    mem2.restore(saved)
    assert mem2.current_turn == mem.current_turn
    assert len(mem2.memories) == len(mem.memories)
    assert mem2.memories[0].memory_id == mem.memories[0].memory_id
    print("[OK] 직렬화/복원 라운드트립 통과")
    print(f"     복원된 기억 수: {len(mem2.memories)}")
