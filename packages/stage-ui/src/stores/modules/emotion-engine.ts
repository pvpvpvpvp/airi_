/**
 * Emotion Engine v2 — 9-theory cognitive science integration
 * ============================================================
 * TypeScript port of emotion_engine_v2.py
 *
 * [기존 4개]
 * 1. 베이즈 뇌 (Bayesian Brain)        — 확률적 감정 업데이트
 * 2. 드리프트 확산 모델 (DDM)           — 결정 지연과 오류
 * 3. 제한된 합리성 (Bounded Rationality) — 만족화 노이즈
 * 4. 마르코프 연쇄 (Markov Chain)       — 감정의 관성
 *
 * [신규 5개]
 * 5. 쾌락 적응 (Hedonic Adaptation)    — 시간/회차에 따른 감정 소멸
 * 6. 평가 이론 (Appraisal Theory)      — 상태 의존적 해석
 * 7. 피크엔드 법칙 (Peak-End Rule)     — 강렬한 기억의 편향
 * 8. 자아 고갈 (Ego Depletion)         — 감정 에너지 소모
 * 9. 인지 부조화 (Cognitive Dissonance) — 감정 충돌 시 불안정
 */

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { SeededRandom } from '../../utils/prng'

// ─────────────────────────────────────────────────
//  감정 & 이벤트 정의
// ─────────────────────────────────────────────────

export enum EngineEmotion {
  Happy = 'happy',
  Neutral = 'neutral',
  Sassy = 'sassy',
  Tired = 'tired',
  Excited = 'excited',
  Confused = 'confused',
}

export const ENGINE_EMOTION_VALUES = Object.values(EngineEmotion)

export const EMOTION_KR: Record<EngineEmotion, string> = {
  [EngineEmotion.Happy]: '기분좋음',
  [EngineEmotion.Neutral]: '평온함',
  [EngineEmotion.Sassy]: '깐찐함',
  [EngineEmotion.Tired]: '귀찮음',
  [EngineEmotion.Excited]: '설렘',
  [EngineEmotion.Confused]: '혼란',
}

export const EMOTION_DIRECTIVES: Record<EngineEmotion, string> = {
  [EngineEmotion.Happy]:
    '지금 기분이 좋다. 밝고 긍정적인 톤으로 대답하고, ㅎㅎ 같은 표현을 자연스럽게 쓴다.',
  [EngineEmotion.Neutral]:
    '지금 평온한 상태다. 담담하고 차분하게 대답한다.',
  [EngineEmotion.Sassy]:
    '지금 좀 깐찐하다. 현명스럽고 아이러니하게 대답하되, 본질적으로는 도움을 주려 한다. \'흠...\' 같은 표현을 쓸 수 있다.',
  [EngineEmotion.Tired]:
    '지금 귀찮고 피곤하다. 대답을 짧게 하려 하고, \'대충\', \'나중에\' 같은 표현을 쓴다.',
  [EngineEmotion.Excited]:
    '지금 매우 설렌다. 대화에 \'대박\', \'헐\' 같은 감탄사를 자연스럽게 쓴다.',
  [EngineEmotion.Confused]:
    '지금 감정적으로 혼란스럽다. 말이 약간 어수선하고, \'잠깐만\', \'뭔지\', \'아닌가...\' 같은 표현을 쓴다. 확신 없이 대답하는 경향이 있다.',
}

// 감정별 추임새 (딜레이 중 표시)
// short: 짧은 지연용, long: 긴 지연용, energy_low: 에너지 부족 시
export const EMOTION_FILLERS: Record<EngineEmotion, { short: string[], long: string[], energy_low: string[] }> = {
  [EngineEmotion.Happy]: {
    short: ['ㅎㅎ', '음~', '잠깐~', '잠깐!', '정말다', '히히'],
    long: [
      '음~ 잠깐 생각 좀 해볼게 ㅎㅎ',
      '음음 잠깐! 좋은 생각 났어',
      '아 이거 재밌는데, 잠깐~',
      'ㅎㅎ 기다려봐 금방 말해줄게',
      '잠깐 이거 좀 생각해볼게!',
    ],
    energy_low: ['ㅎ... 잠깐만', '음 음... 기다려봐'],
  },
  [EngineEmotion.Neutral]: {
    short: ['음...', '잠깐만', '어디 보자', '그게...', '음,'],
    long: [
      '음... 잠깐만 생각 좀 해볼게',
      '어디 보자... 잠깐',
      '그게 맞이야... 잠깐만',
      '음, 잠깐. 정리 좀 하고',
      '음 이거 좀 생각해볼게',
    ],
    energy_low: ['음......', '잠깐...'],
  },
  [EngineEmotion.Sassy]: {
    short: ['흠...', '음 뭔', '어휴', '쯧', '아니', '뭔데'],
    long: [
      '흠... 이걸 내가 왜 생각해야 되는 건데',
      '음 잠깐만 좀. 생각 중이에요',
      '어휴... 기다려봐',
      '모르겠어 잠깐만 좀',
      '쯧... 이것도 내가 해야 돼?',
      '아니 잠깐만 좀 기다려',
    ],
    energy_low: ['흠...... 진짜', '모르겠어... 잠깐...', '어휴......'],
  },
  [EngineEmotion.Tired]: {
    short: ['음...', '일...', '으...', '하아', '어...'],
    long: [
      '음... 잠깐만 좀 쉬고',
      '일... 이거 지금 꼭 해야 돼?',
      '으 잠깐... 머리가 안 돌아가',
      '하아... 생각하기 귀찮은데',
      '어... 대충 해도 돼?',
    ],
    energy_low: [
      '으................',
      '일으......... 잠깐.........',
      '하아..........',
    ],
  },
  [EngineEmotion.Excited]: {
    short: ['오!', '헐!', '잠깐잠깐!', '앗!', '엥!', '오오!'],
    long: [
      '오오오 잠깐만! 이거 진짜 좋은 생각인데!',
      '헐 잠깐! 갑자기 떠오른 게 있어!',
      '앗 잠깐잠깐잠깐! 기다려봐!',
      '아 이거 대박인데 잠깐만!',
      '오! 잠깐만 정리 좀 하고!',
    ],
    energy_low: ['오... 잠깐만 좀..!', '앗! 근데 잠깐...'],
  },
  [EngineEmotion.Confused]: {
    short: ['이...?', '뭔지', '잠깐?', '어?', '어...?', '이라'],
    long: [
      '이...? 잠깐만 뭐였지',
      '어... 아닌가? 잠깐만',
      '어? 음 잠깐, 헷갈려',
      '뭔지... 잠깐만 생각 좀',
      '어...? 잠 아닌가. 잠깐',
      '이라... 잠깐 정리 좀 하고',
    ],
    energy_low: ['이......... 뭐였지.........', '으음...? 잠깐.........'],
  },
}

export enum EngineEvent {
  Praise = 'praise',
  Scold = 'scold',
  Joke = 'joke',
  Ignore = 'ignore',
  AskHard = 'ask_hard',
  AskEasy = 'ask_easy',
  Agree = 'agree',
  Disagree = 'disagree',
}

export const ENGINE_EVENT_VALUES = Object.values(EngineEvent)

export const EVENT_KR: Record<EngineEvent, string> = {
  [EngineEvent.Praise]: '칭찬',
  [EngineEvent.Scold]: '꾸중',
  [EngineEvent.Joke]: '농담',
  [EngineEvent.Ignore]: '무시',
  [EngineEvent.AskHard]: '어려운질문',
  [EngineEvent.AskEasy]: '쉬운질문',
  [EngineEvent.Agree]: '동의',
  [EngineEvent.Disagree]: '반박',
}

// ─────────────────────────────────────────────────
//  BASE_LIKELIHOOD — 정규화 보장 (합 = 1.0)
// ─────────────────────────────────────────────────

const _RAW_LIKELIHOOD: Record<EngineEvent, Record<EngineEmotion, number>> = {
  [EngineEvent.Praise]: { [EngineEmotion.Happy]: 0.45, [EngineEmotion.Neutral]: 0.15, [EngineEmotion.Sassy]: 0.03, [EngineEmotion.Tired]: 0.07, [EngineEmotion.Excited]: 0.40, [EngineEmotion.Confused]: 0.05 },
  [EngineEvent.Scold]: { [EngineEmotion.Happy]: 0.03, [EngineEmotion.Neutral]: 0.10, [EngineEmotion.Sassy]: 0.55, [EngineEmotion.Tired]: 0.12, [EngineEmotion.Excited]: 0.03, [EngineEmotion.Confused]: 0.10 },
  [EngineEvent.Joke]: { [EngineEmotion.Happy]: 0.40, [EngineEmotion.Neutral]: 0.12, [EngineEmotion.Sassy]: 0.08, [EngineEmotion.Tired]: 0.05, [EngineEmotion.Excited]: 0.35, [EngineEmotion.Confused]: 0.05 },
  [EngineEvent.Ignore]: { [EngineEmotion.Happy]: 0.03, [EngineEmotion.Neutral]: 0.12, [EngineEmotion.Sassy]: 0.30, [EngineEmotion.Tired]: 0.45, [EngineEmotion.Excited]: 0.02, [EngineEmotion.Confused]: 0.08 },
  [EngineEvent.AskHard]: { [EngineEmotion.Happy]: 0.08, [EngineEmotion.Neutral]: 0.18, [EngineEmotion.Sassy]: 0.20, [EngineEmotion.Tired]: 0.30, [EngineEmotion.Excited]: 0.08, [EngineEmotion.Confused]: 0.15 },
  [EngineEvent.AskEasy]: { [EngineEmotion.Happy]: 0.30, [EngineEmotion.Neutral]: 0.35, [EngineEmotion.Sassy]: 0.03, [EngineEmotion.Tired]: 0.08, [EngineEmotion.Excited]: 0.22, [EngineEmotion.Confused]: 0.03 },
  [EngineEvent.Agree]: { [EngineEmotion.Happy]: 0.35, [EngineEmotion.Neutral]: 0.25, [EngineEmotion.Sassy]: 0.03, [EngineEmotion.Tired]: 0.08, [EngineEmotion.Excited]: 0.30, [EngineEmotion.Confused]: 0.03 },
  [EngineEvent.Disagree]: { [EngineEmotion.Happy]: 0.03, [EngineEmotion.Neutral]: 0.12, [EngineEmotion.Sassy]: 0.45, [EngineEmotion.Tired]: 0.18, [EngineEmotion.Excited]: 0.05, [EngineEmotion.Confused]: 0.12 },
}

export const BASE_LIKELIHOOD: Record<EngineEvent, Record<EngineEmotion, number>> = {} as Record<EngineEvent, Record<EngineEmotion, number>>
for (const evt of ENGINE_EVENT_VALUES) {
  const raw = _RAW_LIKELIHOOD[evt]
  const total = Object.values(raw).reduce((a, b) => a + b, 0)
  BASE_LIKELIHOOD[evt] = {} as Record<EngineEmotion, number>
  for (const e of ENGINE_EMOTION_VALUES)
    BASE_LIKELIHOOD[evt][e] = raw[e] / total
}

// ─────────────────────────────────────────────────
//  APPRAISAL_MODIFIERS — 48개 조합 (6 상태 × 8 이벤트)
// ─────────────────────────────────────────────────

type AppraisalKey = `${EngineEmotion}:${EngineEvent}`
type EmotionModifiers = Partial<Record<EngineEmotion, number>>

const APPRAISAL_MODIFIERS: Record<AppraisalKey, EmotionModifiers> = {
  // HAPPY
  'happy:praise': { [EngineEmotion.Happy]: 1.3, [EngineEmotion.Excited]: 1.4 },
  'happy:scold': { [EngineEmotion.Sassy]: 1.6, [EngineEmotion.Confused]: 1.5, [EngineEmotion.Happy]: 0.3 },
  'happy:joke': { [EngineEmotion.Happy]: 1.4, [EngineEmotion.Excited]: 1.3 },
  'happy:ignore': { [EngineEmotion.Sassy]: 1.4, [EngineEmotion.Confused]: 1.2 },
  'happy:ask_hard': { [EngineEmotion.Happy]: 0.8, [EngineEmotion.Neutral]: 1.3 },
  'happy:ask_easy': { [EngineEmotion.Happy]: 1.2, [EngineEmotion.Excited]: 1.1 },
  'happy:agree': { [EngineEmotion.Happy]: 1.3, [EngineEmotion.Excited]: 1.2 },
  'happy:disagree': { [EngineEmotion.Confused]: 1.4, [EngineEmotion.Sassy]: 1.3 },
  // NEUTRAL — 변조 최소
  'neutral:praise': { [EngineEmotion.Happy]: 1.1 },
  'neutral:scold': { [EngineEmotion.Sassy]: 1.1 },
  'neutral:joke': { [EngineEmotion.Happy]: 1.1 },
  'neutral:ignore': { [EngineEmotion.Tired]: 1.1 },
  'neutral:ask_hard': { [EngineEmotion.Neutral]: 1.1 },
  'neutral:ask_easy': { [EngineEmotion.Neutral]: 1.1 },
  'neutral:agree': { [EngineEmotion.Happy]: 1.1 },
  'neutral:disagree': { [EngineEmotion.Sassy]: 1.1 },
  // SASSY
  'sassy:praise': { [EngineEmotion.Happy]: 1.8, [EngineEmotion.Sassy]: 0.3 },
  'sassy:scold': { [EngineEmotion.Sassy]: 1.6, [EngineEmotion.Tired]: 1.3 },
  'sassy:joke': { [EngineEmotion.Sassy]: 1.5, [EngineEmotion.Happy]: 0.5 },
  'sassy:ignore': { [EngineEmotion.Sassy]: 1.4, [EngineEmotion.Tired]: 1.3 },
  'sassy:ask_hard': { [EngineEmotion.Sassy]: 1.5, [EngineEmotion.Tired]: 1.4 },
  'sassy:ask_easy': { [EngineEmotion.Sassy]: 0.7, [EngineEmotion.Neutral]: 1.3 },
  'sassy:agree': { [EngineEmotion.Happy]: 1.4, [EngineEmotion.Sassy]: 0.5 },
  'sassy:disagree': { [EngineEmotion.Sassy]: 1.7, [EngineEmotion.Tired]: 1.2 },
  // TIRED
  'tired:praise': { [EngineEmotion.Happy]: 1.5, [EngineEmotion.Tired]: 0.6 },
  'tired:scold': { [EngineEmotion.Tired]: 1.3, [EngineEmotion.Sassy]: 1.4 },
  'tired:joke': { [EngineEmotion.Tired]: 1.2, [EngineEmotion.Happy]: 0.7 },
  'tired:ignore': { [EngineEmotion.Tired]: 1.5 },
  'tired:ask_hard': { [EngineEmotion.Tired]: 1.8, [EngineEmotion.Sassy]: 1.4 },
  'tired:ask_easy': { [EngineEmotion.Tired]: 0.8, [EngineEmotion.Neutral]: 1.3 },
  'tired:agree': { [EngineEmotion.Tired]: 0.7, [EngineEmotion.Happy]: 1.3 },
  'tired:disagree': { [EngineEmotion.Tired]: 1.3, [EngineEmotion.Sassy]: 1.5 },
  // EXCITED
  'excited:praise': { [EngineEmotion.Excited]: 1.5, [EngineEmotion.Happy]: 1.3 },
  'excited:scold': { [EngineEmotion.Confused]: 1.6, [EngineEmotion.Sassy]: 1.3, [EngineEmotion.Excited]: 0.3 },
  'excited:joke': { [EngineEmotion.Excited]: 1.4, [EngineEmotion.Happy]: 1.3 },
  'excited:ignore': { [EngineEmotion.Sassy]: 1.7, [EngineEmotion.Confused]: 1.3, [EngineEmotion.Excited]: 0.3 },
  'excited:ask_hard': { [EngineEmotion.Excited]: 0.6, [EngineEmotion.Neutral]: 1.3 },
  'excited:ask_easy': { [EngineEmotion.Excited]: 1.3, [EngineEmotion.Happy]: 1.2 },
  'excited:agree': { [EngineEmotion.Excited]: 1.4, [EngineEmotion.Happy]: 1.2 },
  'excited:disagree': { [EngineEmotion.Confused]: 1.5, [EngineEmotion.Sassy]: 1.3 },
  // CONFUSED
  'confused:praise': { [EngineEmotion.Happy]: 1.3, [EngineEmotion.Confused]: 0.6 },
  'confused:scold': { [EngineEmotion.Confused]: 1.4, [EngineEmotion.Sassy]: 1.3 },
  'confused:joke': { [EngineEmotion.Confused]: 1.3, [EngineEmotion.Happy]: 0.8 },
  'confused:ignore': { [EngineEmotion.Confused]: 1.3, [EngineEmotion.Tired]: 1.3 },
  'confused:ask_hard': { [EngineEmotion.Confused]: 1.5, [EngineEmotion.Tired]: 1.3 },
  'confused:ask_easy': { [EngineEmotion.Confused]: 0.5, [EngineEmotion.Neutral]: 1.5 },
  'confused:agree': { [EngineEmotion.Happy]: 1.6, [EngineEmotion.Confused]: 0.4 },
  'confused:disagree': { [EngineEmotion.Confused]: 1.6, [EngineEmotion.Sassy]: 1.2 },
}

// ─────────────────────────────────────────────────
//  [4] 마르코프 전이 행렬 — 감정 간 자연스러운 '거리' 반영
//  행(row) = 현재 감정, 열(col) = 다음 감정으로의 전이 선호도 (각 행 합 = 1.0)
//  설계 원칙:
//    - 감정 유사성: Happy↔Excited, Sassy↔Tired 는 전이 확률 높음
//    - 감정 반발성: Happy→Tired, Excited→Sassy 는 전이 확률 낮음
//    - Neutral은 모든 감정에서 중간 수준의 탈출구 역할
// ─────────────────────────────────────────────────

const MARKOV_TRANSITION: Record<EngineEmotion, Record<EngineEmotion, number>> = {
  [EngineEmotion.Happy]: {
    [EngineEmotion.Happy]: 0.50, [EngineEmotion.Neutral]: 0.20, [EngineEmotion.Sassy]: 0.05,
    [EngineEmotion.Tired]: 0.05, [EngineEmotion.Excited]: 0.15, [EngineEmotion.Confused]: 0.05,
  },
  [EngineEmotion.Neutral]: {
    [EngineEmotion.Happy]: 0.15, [EngineEmotion.Neutral]: 0.50, [EngineEmotion.Sassy]: 0.10,
    [EngineEmotion.Tired]: 0.10, [EngineEmotion.Excited]: 0.10, [EngineEmotion.Confused]: 0.05,
  },
  [EngineEmotion.Sassy]: {
    [EngineEmotion.Happy]: 0.05, [EngineEmotion.Neutral]: 0.15, [EngineEmotion.Sassy]: 0.45,
    [EngineEmotion.Tired]: 0.20, [EngineEmotion.Excited]: 0.03, [EngineEmotion.Confused]: 0.12,
  },
  [EngineEmotion.Tired]: {
    [EngineEmotion.Happy]: 0.05, [EngineEmotion.Neutral]: 0.20, [EngineEmotion.Sassy]: 0.25,
    [EngineEmotion.Tired]: 0.40, [EngineEmotion.Excited]: 0.02, [EngineEmotion.Confused]: 0.08,
  },
  [EngineEmotion.Excited]: {
    [EngineEmotion.Happy]: 0.25, [EngineEmotion.Neutral]: 0.15, [EngineEmotion.Sassy]: 0.05,
    [EngineEmotion.Tired]: 0.15, [EngineEmotion.Excited]: 0.35, [EngineEmotion.Confused]: 0.05,
  },
  [EngineEmotion.Confused]: {
    [EngineEmotion.Happy]: 0.05, [EngineEmotion.Neutral]: 0.20, [EngineEmotion.Sassy]: 0.20,
    [EngineEmotion.Tired]: 0.20, [EngineEmotion.Excited]: 0.03, [EngineEmotion.Confused]: 0.32,
  },
}

// ─────────────────────────────────────────────────
//  결과 타입
// ─────────────────────────────────────────────────

export interface EmotionEngineResult {
  state: EngineEmotion
  probabilities: Record<EngineEmotion, number>
  prevState: EngineEmotion
  directive: string
  ddmSteps: number
  ddmDelayMs: number
  energy: number
  dissonance: number
  peakMemory: { emotion: EngineEmotion, intensity: number, turn: number } | null
  decayApplied: boolean
  filler: string
}

// PATCH-3: 직렬화 가능 상태 (lastEventTime 제외 — 복원 시 Date.now()로 리셋)
export interface EmotionEngineSerializable {
  currentState: EngineEmotion
  turnCount: number
  conversationTurnCount: number
  energy: number
  dissonance: number
  prior: Record<EngineEmotion, number>
  peakMemory: Array<{ emotion: EngineEmotion, intensity: number, turn: number }>
  recentEvents: EngineEvent[]
  turnsSinceStrongEvent: number
  rngState: [number, number, number, number]
}

// ─────────────────────────────────────────────────
//  감정 이벤트 지시 — 정적 상수 (시스템 프롬프트 base에 1회만 포함)
// ─────────────────────────────────────────────────

export const EMOTION_ENGINE_EVENT_INSTRUCTIONS
  = '\n\n[감정 이벤트 지시]'
  + '\n대화 중 아래 이벤트를 감지하면 반드시 해당 ACT 토큰을 응답에 포함하라.'
  + '\n형식: <|ACT: {"event": "<이벤트명>"}|>'
  + '\n- praise: 상대가 나를 칭찬하거나 긍정적으로 평가할 때'
  + '\n- scold: 상대가 나를 꾸짖거나 부정적으로 비판할 때'
  + '\n- joke: 농담이나 유머러스한 상호작용이 있을 때'
  + '\n- ignore: 내 말이 무시되거나 대화가 단절될 때'
  + '\n- ask_hard: 어렵거나 복잡한 질문을 받을 때'
  + '\n- ask_easy: 간단하고 쉬운 질문을 받을 때'
  + '\n- agree: 상대가 내 의견에 동의하거나 공감할 때'
  + '\n- disagree: 상대가 내 의견에 반박하거나 이의를 제기할 때'
  + '\n(기존 <|ACT: {"emotion": "..."}|> 형식도 병행 사용 가능)'

// ─────────────────────────────────────────────────
//  내부 헬퍼
// ─────────────────────────────────────────────────

function uniformPrior(): Record<EngineEmotion, number> {
  const n = ENGINE_EMOTION_VALUES.length
  return Object.fromEntries(ENGINE_EMOTION_VALUES.map(e => [e, 1 / n])) as Record<EngineEmotion, number>
}

function normalize(probs: Record<EngineEmotion, number>): Record<EngineEmotion, number> {
  const total = Object.values(probs).reduce((a, b) => a + b, 0)
  return Object.fromEntries(
    Object.entries(probs).map(([k, v]) => [k, v / total]),
  ) as Record<EngineEmotion, number>
}

// ─────────────────────────────────────────────────
//  감정 엔진 Pinia Store
// ─────────────────────────────────────────────────

export const useEmotionEngineStore = defineStore('emotion-engine', () => {
  // ── 설정 파라미터 ──
  const inertia = 0.20
  const laziness = 0.10
  const ddmNoise = 0.08
  const ddmDriftRate = 0.15
  const decayRate = 0.15
  const decayTurnInterval = 3
  const energyDrain = 0.06
  const energyRegen = 0.08
  const peakWeight = 0.15
  const dissonanceThreshold = 0.55

  // PATCH-4: 시드 기반 PRNG — Math.random() 대신 사용
  const rng = ref(new SeededRandom())

  // ── 상태 ──
  const currentState = ref<EngineEmotion>(EngineEmotion.Neutral)
  const turnCount = ref(0)
  // 실제 대화 회차 — processEvent와 setEmotionDirect 양쪽에서 모두 증가
  // turnCount는 engine-event 경로에서만 증가하므로 buildEmotionPrompt용으로 부정확함
  const conversationTurnCount = ref(0)
  const energy = ref(1.0)
  const prior = ref<Record<EngineEmotion, number>>(uniformPrior())
  const peakMemory = ref<Array<{ emotion: EngineEmotion, intensity: number, turn: number }>>([])
  const recentEvents = ref<EngineEvent[]>([])
  const dissonance = ref(0.0)
  const turnsSinceStrongEvent = ref(0)

  // ── computed ──
  const directive = computed(() => EMOTION_DIRECTIVES[currentState.value])
  const emotionKr = computed(() => EMOTION_KR[currentState.value])

  // ── [5] 쾌락 적응: turn 기반 decay ──
  function applyDecay(): boolean {
    turnsSinceStrongEvent.value += 1
    if (turnsSinceStrongEvent.value < decayTurnInterval)
      return false

    // 카운터 리셋: decayTurnInterval마다 1회만 실행되도록 보장
    turnsSinceStrongEvent.value = 0

    const maxProb = Math.max(...Object.values(prior.value))
    // NOTICE: maxProb에 비례하도록 수정 — 감정이 지배적일수록 decay를 강하게 적용해야
    // lock-in을 방지할 수 있다. 기존 (1-maxProb) 공식은 역방향이었음.
    const intensityFactor = Math.max(0.3, maxProb)
    const decayFactor = Math.min(0.6, decayRate * intensityFactor)
    const baseline = 1.0 / ENGINE_EMOTION_VALUES.length

    const next = { ...prior.value } as Record<EngineEmotion, number>
    for (const e of ENGINE_EMOTION_VALUES)
      next[e] = next[e] * (1 - decayFactor) + baseline * decayFactor
    prior.value = next
    return true
  }

  // ── [6] 평가 이론 ──
  function appraise(event: EngineEvent): Record<EngineEmotion, number> {
    const lik = { ...BASE_LIKELIHOOD[event] } as Record<EngineEmotion, number>
    const key: AppraisalKey = `${currentState.value}:${event}`
    const mods = APPRAISAL_MODIFIERS[key]
    if (mods) {
      for (const [emotion, multiplier] of Object.entries(mods) as [EngineEmotion, number][])
        lik[emotion] = (lik[emotion] ?? 0.01) * multiplier
    }
    return normalize(lik)
  }

  // ── [1] 베이즈 업데이트 ──
  function bayesUpdate(likelihood: Record<EngineEmotion, number>): Record<EngineEmotion, number> {
    const pEvidence = ENGINE_EMOTION_VALUES.reduce(
      (acc, e) => acc + likelihood[e] * prior.value[e],
      0,
    )
    const posterior = Object.fromEntries(
      ENGINE_EMOTION_VALUES.map(e => [e, (likelihood[e] * prior.value[e]) / pEvidence]),
    ) as Record<EngineEmotion, number>
    prior.value = { ...posterior }
    return posterior
  }

  // ── [7] 피크엔드 기록 ──
  function recordPeak(posterior: Record<EngineEmotion, number>): void {
    const maxEmotion = ENGINE_EMOTION_VALUES.reduce((a, b) => posterior[a] > posterior[b] ? a : b)
    const intensity = posterior[maxEmotion]
    if (intensity > 0.5) {
      // conversationTurnCount 기준으로 저장 — 피크 recency 계산에 사용
      peakMemory.value.push({ emotion: maxEmotion, intensity, turn: conversationTurnCount.value })
      if (peakMemory.value.length > 5)
        peakMemory.value = peakMemory.value.slice(-5)
      turnsSinceStrongEvent.value = 0
    }
  }

  // ── [7] 피크엔드 편향 ──
  function applyPeakMemory(posterior: Record<EngineEmotion, number>): Record<EngineEmotion, number> {
    if (peakMemory.value.length === 0)
      return posterior

    const peak = peakMemory.value.reduce((a, b) => a.intensity > b.intensity ? a : b)
    const last = peakMemory.value[peakMemory.value.length - 1]

    const biased = { ...posterior }
    biased[peak.emotion] += peakWeight * peak.intensity
    biased[last.emotion] += peakWeight * 0.5 * last.intensity

    return normalize(biased)
  }

  // ── [4] 마르코프 전이 행렬 ──
  function markovBlend(posterior: Record<EngineEmotion, number>): Record<EngineEmotion, number> {
    // 전이 행렬에서 현재 감정의 선호 분포를 꺼내 inertia 비율로 블렌딩
    // 기존: 현재 감정에만 +inertia (1차원 관성)
    // 개선: 전이 행렬로 감정 간 거리(친밀/반발) 반영
    const transition = MARKOV_TRANSITION[currentState.value as EngineEmotion]
    const blended = {} as Record<EngineEmotion, number>
    for (const e of ENGINE_EMOTION_VALUES)
      blended[e] = posterior[e] * (1 - inertia) + transition[e] * inertia
    return normalize(blended)
  }

  // ── [8] 에너지 업데이트 ──
  const DRAIN_EVENTS: Partial<Record<EngineEvent, number>> = {
    [EngineEvent.AskHard]: 2.0,
    [EngineEvent.Scold]: 1.5,
    [EngineEvent.Disagree]: 1.5,
    [EngineEvent.Ignore]: 1.2,
  }
  const NEUTRAL_DRAIN_EVENTS: Partial<Record<EngineEvent, number>> = {
    [EngineEvent.AskEasy]: 0.5,
    [EngineEvent.Joke]: 0.3,
  }
  const REGEN_EVENTS: Partial<Record<EngineEvent, number>> = {
    [EngineEvent.Praise]: 2.0,
    [EngineEvent.Agree]: 1.5,
  }

  function updateEnergy(event: EngineEvent): void {
    // 점근적 자동 회복: 에너지가 낮을수록 회복 빠르고, 가득 찰수록 느림
    // += 0.01 * (1 - energy): 1.0 근방에서 자연스럽게 수렴 (HP포션식 직선 → 피로 곡선)
    energy.value = Math.min(1.0, energy.value + 0.01 * (1.0 - energy.value))

    // Tired 상태에서 Joke/Praise → 1.5x 감동 보정
    const tiredBonus = currentState.value === EngineEmotion.Tired
      && (event === EngineEvent.Joke || event === EngineEvent.Praise)

    if (event in DRAIN_EVENTS) {
      energy.value = Math.max(0.0, energy.value - energyDrain * DRAIN_EVENTS[event]!)
    }
    else if (event in REGEN_EVENTS) {
      const mult = REGEN_EVENTS[event]! * (tiredBonus ? 1.5 : 1.0)
      energy.value = Math.min(1.0, energy.value + energyRegen * mult)
    }
    else if (event in NEUTRAL_DRAIN_EVENTS) {
      if (tiredBonus) {
        // Joke while Tired → drain 대신 1.5x 회복으로 전환
        energy.value = Math.min(1.0, energy.value + energyRegen * 1.5)
      }
      else {
        energy.value = Math.max(0.0, energy.value - energyDrain * NEUTRAL_DRAIN_EVENTS[event]!)
      }
    }
  }

  // ── [3+8] 만족화 ──
  function satisfice(probs: Record<EngineEmotion, number>): Record<EngineEmotion, number> {
    const energyMultiplier = 1.0 + 2.0 * (1.0 - energy.value)
    const noiseScale = laziness * energyMultiplier * (0.8 + rng.value.random() * 0.4)

    const noisy = {} as Record<EngineEmotion, number>
    for (const e of ENGINE_EMOTION_VALUES)
      noisy[e] = Math.max(0.01, probs[e] + (rng.value.random() - 0.5) * noiseScale)

    if (energy.value < 0.3) {
      noisy[EngineEmotion.Tired] *= 1.5
      noisy[EngineEmotion.Sassy] *= 1.3
    }
    return normalize(noisy)
  }

  // ── [9] 인지 부조화 ──
  const CONFLICT_PAIRS: [Set<EngineEvent>, number][] = [
    [new Set([EngineEvent.Praise, EngineEvent.Scold]), 0.8],
    [new Set([EngineEvent.Agree, EngineEvent.Disagree]), 0.7],
    [new Set([EngineEvent.Joke, EngineEvent.Scold]), 0.5],
    [new Set([EngineEvent.Praise, EngineEvent.Ignore]), 0.6],
    [new Set([EngineEvent.Agree, EngineEvent.Scold]), 0.5],
    [new Set([EngineEvent.Praise, EngineEvent.Disagree]), 0.5],
    [new Set([EngineEvent.Joke, EngineEvent.Ignore]), 0.4],
  ]

  function computeDissonance(event: EngineEvent): number {
    recentEvents.value.push(event)
    if (recentEvents.value.length > 5)
      recentEvents.value = recentEvents.value.slice(-5)

    if (recentEvents.value.length < 2) {
      dissonance.value = 0.0
      return 0.0
    }

    // recentEvents는 최대 5개로 유지되므로 전체 윈도우를 사용
    const recent = recentEvents.value
    let totalConflict = 0.0
    for (let i = 0; i < recent.length; i++) {
      for (let j = i + 1; j < recent.length; j++) {
        const pair = new Set([recent[i], recent[j]])
        for (const [conflictPair, weight] of CONFLICT_PAIRS) {
          if (pair.size === conflictPair.size && [...pair].every(v => conflictPair.has(v))) {
            const recency = 1.0 - (j - i) * 0.2
            totalConflict += weight * recency
            break
          }
        }
      }
    }

    dissonance.value = Math.min(1.0, dissonance.value * 0.6 + totalConflict)
    return dissonance.value
  }

  // ── [9] 부조화 → 분포 반영 ──
  function applyDissonanceToProbs(probs: Record<EngineEmotion, number>): Record<EngineEmotion, number> {
    if (dissonance.value <= dissonanceThreshold)
      return probs

    const boost = dissonance.value * 0.25
    const adjusted = { ...probs }
    adjusted[EngineEmotion.Confused] += boost
    return normalize(adjusted)
  }

  // ── [2] DDM ──
  function ddmDecide(probs: Record<EngineEmotion, number>): { winner: EngineEmotion, steps: number, delayMs: number } {
    const baseThreshold = 0.3 + rng.value.random() * 0.2
    const dissonanceWobble = dissonance.value * (rng.value.random() - 0.5) * 0.1
    const threshold = Math.max(0.15, baseThreshold + dissonanceWobble)

    const accum = Object.fromEntries(ENGINE_EMOTION_VALUES.map(e => [e, 0.0])) as Record<EngineEmotion, number>
    let steps = 0
    let winner: EngineEmotion | null = null

    while (winner === null && steps < 60) {
      steps++
      for (const e of ENGINE_EMOTION_VALUES) {
        const drift = probs[e] * ddmDriftRate
        const noise = (rng.value.random() - 0.5) * ddmNoise
        accum[e] += drift + noise
      }
      // NOTICE: winner 체크를 inner loop 밖으로 이동 — 기존 코드는 같은 step에서
      // 여러 감정이 threshold를 넘을 때 배열 순서(happy 우선)로 당선자가 결정되는
      // 순서 편향이 있었음. 이제 한 step 내 모든 누적 완료 후 가장 높은 값을 당선자로 선택.
      const crossed = ENGINE_EMOTION_VALUES.filter(e => accum[e] >= threshold)
      if (crossed.length > 0)
        winner = crossed.reduce((a, b) => accum[a] > accum[b] ? a : b)
    }

    if (winner === null)
      winner = ENGINE_EMOTION_VALUES.reduce((a, b) => accum[a] > accum[b] ? a : b)

    // PATCH-2: delay_ms 최대 1200ms 캡 — LLM 응답 대기 중 인위적 지연 방지
    const rawDelay = Math.floor(300 + steps * 70 + rng.value.random() * 200 + dissonance.value * 300)
    const delayMs = Math.min(rawDelay, 1200)

    return { winner, steps, delayMs }
  }

  // ── 추임새 선택 ──
  function pickFiller(state: EngineEmotion, delayMs: number): string {
    const fillers = EMOTION_FILLERS[state]
    let pool: string[]
    if (energy.value < 0.3 && fillers.energy_low.length > 0)
      pool = fillers.energy_low
    else if (delayMs > 800)
      pool = fillers.long
    else
      pool = fillers.short

    return pool[Math.floor(rng.value.random() * pool.length)]
  }

  // ─────────────────────────────────────────────
  //  통합 파이프라인
  // ─────────────────────────────────────────────

  function processEvent(event: EngineEvent | string): EmotionEngineResult {
    if (!ENGINE_EVENT_VALUES.includes(event as EngineEvent)) {
      console.warn(`[EmotionEngine] Unknown event "${event}", skipping.`)
      return {
        state: currentState.value,
        probabilities: { ...prior.value },
        prevState: currentState.value,
        directive: EMOTION_DIRECTIVES[currentState.value],
        ddmSteps: 0,
        ddmDelayMs: 0,
        energy: energy.value,
        dissonance: dissonance.value,
        peakMemory: peakMemory.value.at(-1) ?? null,
        decayApplied: false,
        filler: '',
      }
    }
    const evt = event as EngineEvent
    turnCount.value++
    conversationTurnCount.value++
    const prev = currentState.value

    // [5] 쾌락 적응
    const decayApplied = applyDecay()

    // [8] 에너지
    updateEnergy(evt)

    // [9] 부조화
    computeDissonance(evt)

    // [6] 평가
    const likelihood = appraise(evt)

    // [1] 베이즈
    let posterior = bayesUpdate(likelihood)

    // [7] 피크 기록 (posterior 기준)
    recordPeak(posterior)

    // [7] 피크엔드 편향
    posterior = applyPeakMemory(posterior)

    // [4] 마르코프
    let blended = markovBlend(posterior)

    // [3+8] 만족화
    blended = satisfice(blended)

    // [9] 부조화 → 분포 반영
    blended = applyDissonanceToProbs(blended)

    // [2] DDM
    const { winner, steps, delayMs } = ddmDecide(blended)
    currentState.value = winner

    const filler = pickFiller(winner, delayMs)

    const result: EmotionEngineResult = {
      state: winner,
      probabilities: blended,
      prevState: prev,
      directive: EMOTION_DIRECTIVES[winner],
      ddmSteps: steps,
      ddmDelayMs: delayMs,
      energy: energy.value,
      dissonance: dissonance.value,
      peakMemory: peakMemory.value.length > 0
        ? peakMemory.value[peakMemory.value.length - 1]
        : null,
      decayApplied,
      filler,
    }

    _persistState()
    return result
  }

  // PATCH-5: 패스스루 모드 — 기존 emotion ACT 하위 호환
  // 엔진 파이프라인(베이즈/DDM 등)을 우회하고 상태만 직접 설정.
  // turnCount를 올리지 않는 것이 의도적임 — 패스스루는 "엔진 이벤트"가 아니라
  // 외부 명령이므로 decay 카운터에 영향을 주지 않는다.
  function setEmotionDirect(emotion: EngineEmotion | string): EmotionEngineResult {
    const emo = emotion as EngineEmotion
    const prev = currentState.value
    currentState.value = emo
    conversationTurnCount.value++

    // prior를 해당 감정 중심으로 soft-set (급격한 전환 방지)
    const n = ENGINE_EMOTION_VALUES.length
    for (const e of ENGINE_EMOTION_VALUES)
      prior.value[e] = e === emo ? 0.6 : 0.4 / (n - 1)

    // 에너지/부조화는 건드리지 않음 (연속성 유지)

    const result: EmotionEngineResult = {
      state: emo,
      probabilities: { ...prior.value },
      prevState: prev,
      directive: EMOTION_DIRECTIVES[emo],
      ddmSteps: 0,
      ddmDelayMs: 0,
      energy: energy.value,
      dissonance: dissonance.value,
      peakMemory: peakMemory.value.at(-1) ?? null,
      decayApplied: false,
      filler: '',
    }

    _persistState()
    return result
  }

  // ─────────────────────────────────────────────
  //  시스템 프롬프트 빌더
  // ─────────────────────────────────────────────

  function buildEmotionPrompt(): string {
    let energyDesc = ''
    if (energy.value < 0.3)
      energyDesc = '\n감정 에너지가 바닥나 짜증이 나기 쉬운 상태다.'
    else if (energy.value < 0.6)
      energyDesc = '\n감정 에너지가 조금 부족해 평소보다 덜 참을성이 있다.'

    let dissonanceDesc = ''
    if (dissonance.value > 0.5) {
      dissonanceDesc
        = '\n최근 상반된 반응에 감정적으로 혼란스럽다. '
        + '확신 없이 말하는 경향이 있다.'
    }

    let peakDesc = ''
    if (peakMemory.value.length > 0) {
      type PeakEntry = { emotion: EngineEmotion, intensity: number, turn: number }
      const peak = peakMemory.value.reduce(
        (a: PeakEntry, b: PeakEntry) => a.intensity > b.intensity ? a : b,
      )
      // 피크가 15턴 이내일 때만 프롬프트에 포함 — 오래된 기억은 자연스럽게 소멸
      const turnsAgo = conversationTurnCount.value - peak.turn
      if (turnsAgo <= 15)
        peakDesc = `\n과거 대화에서 강하게 '${EMOTION_KR[peak.emotion as EngineEmotion]}' 감정을 느꼈던 기억이 남아있다.`
    }

    return (
      `\n\n[감정 상태: ${EMOTION_KR[currentState.value]}]`
      + `\n${EMOTION_DIRECTIVES[currentState.value]}`
      + energyDesc
      + dissonanceDesc
      + peakDesc
      + `\n(대화 회차: ${conversationTurnCount.value}회차 | 에너지: ${Math.round(energy.value * 100)}%)`
    )
  }

  // PATCH-3: 직렬화 — sessionStorage 저장용
  function serialize(): EmotionEngineSerializable {
    return {
      currentState: currentState.value,
      turnCount: turnCount.value,
      conversationTurnCount: conversationTurnCount.value,
      energy: energy.value,
      dissonance: dissonance.value,
      prior: { ...prior.value },
      peakMemory: [...peakMemory.value],
      recentEvents: [...recentEvents.value],
      turnsSinceStrongEvent: turnsSinceStrongEvent.value,
      rngState: rng.value.getState(),
    }
  }

  // PATCH-3: 복원 — 페이지 리로드/HMR 후 상태 이어받기
  function restore(data: EmotionEngineSerializable): void {
    currentState.value = data.currentState
    turnCount.value = data.turnCount
    // 구버전 sessionStorage에 conversationTurnCount가 없을 수 있으므로 fallback
    conversationTurnCount.value = data.conversationTurnCount ?? data.turnCount
    energy.value = data.energy
    dissonance.value = data.dissonance
    prior.value = { ...data.prior }
    peakMemory.value = [...data.peakMemory]
    recentEvents.value = [...data.recentEvents]
    turnsSinceStrongEvent.value = data.turnsSinceStrongEvent
    rng.value.setState(data.rngState)
    // lastEventTime은 복원하지 않음 — 시간 기반 decay를 위해 Date.now()로 리셋
  }

  // PATCH-3: processEvent/setEmotionDirect 이후 호출되는 내부 저장 트리거
  // watch(turnCount) 대신 명시적으로 호출 — passthrough 경로 누락 방지
  function _persistState(): void {
    try {
      sessionStorage.setItem('emotion-engine-state', JSON.stringify(serialize()))
    }
    catch {
      // quota 초과 시 무시
    }
  }

  // PATCH-3: 페이지 로드 시 sessionStorage에서 복원 시도
  function tryRestoreFromSession(): boolean {
    try {
      const raw = sessionStorage.getItem('emotion-engine-state')
      if (!raw)
        return false
      const data = JSON.parse(raw) as EmotionEngineSerializable
      restore(data)
      return true
    }
    catch {
      return false
    }
  }

  // PATCH-4: 시드 초기화 — 테스트 재현성 / A/B 비교용
  function init(options?: { seed?: number }): void {
    if (options?.seed !== undefined)
      rng.value = new SeededRandom(options.seed)
    else
      rng.value = new SeededRandom()
  }

  function reset(): void {
    currentState.value = EngineEmotion.Neutral
    turnCount.value = 0
    conversationTurnCount.value = 0
    energy.value = 1.0
    dissonance.value = 0.0
    peakMemory.value = []
    recentEvents.value = []
    turnsSinceStrongEvent.value = 0
    prior.value = uniformPrior()
    rng.value = new SeededRandom()
  }

  return {
    // state
    currentState,
    turnCount,
    conversationTurnCount,
    energy,
    dissonance,
    prior,
    peakMemory,
    // computed
    directive,
    emotionKr,
    // methods
    processEvent,
    setEmotionDirect,
    buildEmotionPrompt,
    serialize,
    restore,
    tryRestoreFromSession,
    init,
    reset,
  }
})
