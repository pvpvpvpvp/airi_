/**
 * Belief & Value Engine — 신념 및 가치관 엔진
 * ================================================================
 * emotion-engine.ts, memory-engine.ts와 연동하여 동작하는 가치관 기반 에이전트 엔진.
 * 사용자 발화에서 가치관을 실시간 추출하고, 페르소나 벡터와의 코사인 유사도로
 * 인지 부조화를 수치화하여 BDI 아키텍처 기반 의도 전환을 수행한다.
 *
 * [이론 기반]
 * 1. Schwartz 인간 기본 가치 이론 (STBV, 2012)         — 10차원 가치관 벡터 페르소나
 * 2. 도덕 기반 이론 (MFT, Haidt & Joseph, 2012)        — 6차원 도덕 직관 벡터
 * 3. BDI 아키텍처 (Rao & Georgeff, 1991)                — Belief-Desire-Intention 의사결정
 * 4. 경량 어휘 기반 추출 (PVD / MFD 2.0 영감)           — O(N) 실시간 가치관 키워드 추출
 * 5. 가중치 코사인 유사도 부조화 연산                    — 차원별 민감도 행렬(W_sens) 포함
 * 6. 지수 이동 평균 누적 부조화                          — CD(t) = γ·CD(t-1) + α·D(t)
 *
 * [BDI 의도 전이]
 * COOPERATE → (D_value ≥ DISSONANCE_THRESHOLD) → PERSUASION
 * COOPERATE → (D_value ≥ HIGH_DISSONANCE or CD > CUMULATIVE_THRESHOLD) → REBUTTAL
 * REBUTTAL / PERSUASION → (D_value < threshold) → COOPERATE
 *
 * [감정 엔진 연동]
 * REBUTTAL 진입 시 Stage.vue를 통해 emotionEngineStore.setEmotionDirect('sassy') 호출.
 *
 * [프롬프트 연동]
 * airi-card.ts systemPrompt computed에서 buildBeliefDirective()를 이어 붙임.
 * COOPERATE 상태에서는 빈 문자열 반환 — 프롬프트 캐시 오염 방지.
 */

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

// ─────────────────────────────────────────────────
//  타입 정의
// ─────────────────────────────────────────────────

/** Schwartz 인간 기본 가치 이론 (STBV) 10차원 */
export enum STBVDimension {
  SelfDirection = 'self_direction',
  Stimulation   = 'stimulation',
  Hedonism      = 'hedonism',
  Achievement   = 'achievement',
  Power         = 'power',
  Security      = 'security',
  Conformity    = 'conformity',
  Tradition     = 'tradition',
  Benevolence   = 'benevolence',
  Universalism  = 'universalism',
}

/** 도덕 기반 이론 (MFT) 6차원 */
export enum MFTDimension {
  Care      = 'care',
  Fairness  = 'fairness',
  Loyalty   = 'loyalty',
  Authority = 'authority',
  Sanctity  = 'sanctity',
  Liberty   = 'liberty',
}

/** BDI 의도 상태 */
export enum Intention {
  Cooperate  = 'cooperate',   // 협력 — 기본 상태
  CasualChat = 'casual_chat', // 일상 대화 (가치관 무관 영역)
  Persuasion = 'persuasion',  // 설득 — 중간 수준 부조화
  Rebuttal   = 'rebuttal',    // 반박 — 고강도 부조화 (아부성 차단 지시어 발동)
  Refusal    = 'refusal',     // 단호한 거절 — 극단적 부조화 (향후 확장)
}

export type STBVVector = Record<STBVDimension, number>  // -1.0 ~ 1.0
export type MFTVector  = Record<MFTDimension,  number>  // -1.0 ~ 1.0

export interface BeliefPersonaConfig {
  stbv?:            Partial<STBVVector>
  mft?:             Partial<MFTVector>
  sensitivitySTBV?: Partial<STBVVector>  // 차원별 민감도 가중치 (기본값 1.0)
  sensitivityMFT?:  Partial<MFTVector>
}

export interface BDIAnalysisResult {
  intention:            Intention
  intentionChanged:     boolean
  currentDissonance:    number
  cumulativeDissonance: number
  conflictDimension:    string | null
  shouldSpikeEmotion:   boolean
  spikeEmotion:         'sassy' | 'confused' | null
}

// ─────────────────────────────────────────────────
//  하이퍼파라미터
// ─────────────────────────────────────────────────

// BDI 의도 전환 임계값
const DISSONANCE_THRESHOLD  = 0.65  // D_value ≥ 이 값 → PERSUASION
const HIGH_DISSONANCE       = 0.85  // D_value ≥ 이 값 → REBUTTAL + 감정 스파이크
const CUMULATIVE_THRESHOLD  = 1.5   // 누적 CD > 이 값 → REBUTTAL (논문 기준값)

// 지수 이동 평균 계수
const DECAY_FACTOR  = 0.7  // γ — 이전 누적 부조화 감쇠율
const ALPHA         = 1.0  // α — 현재 턴 부조화 가중치

// ─────────────────────────────────────────────────
//  경량 어휘 사전 (PVD / MFD 2.0 대표 부분 집합)
//  NOTICE: 완전한 PVD(1,068단어)와 MFD 2.0(210+단어/차원)에서 영감을 받은
//          대표 키워드 부분 집합. 영어 + 한국어 어간 형태 병행 수록.
//          형태소 분석기 없이 단순 부분 문자열(substring) 탐색 — O(N).
//          한국어 어간 수록으로 조사·어미 변화를 부분적으로 포괄함.
// ─────────────────────────────────────────────────

interface DimensionLexicon {
  pos: readonly string[]  // 해당 차원을 긍정·강화하는 키워드
  neg: readonly string[]  // 해당 차원을 부정·공격하는 키워드
}

const STBV_LEXICON: Record<STBVDimension, DimensionLexicon> = {
  [STBVDimension.SelfDirection]: {
    pos: [
      'freedom', 'free', 'independent', 'autonomy', 'autonomous', 'creative', 'creativity',
      'curious', 'curiosity', 'explore', 'choose', 'choice', 'self-reliant', 'initiative',
      '자유', '독립', '자율', '창의', '선택', '주체', '자기결정', '탐구', '자립',
    ],
    neg: [
      'control', 'restrict', 'mandate', 'force', 'dictate', 'obey', 'comply', 'censor',
      '통제', '억압', '강요', '명령', '복종', '검열', '제한', '구속', '강제',
    ],
  },
  [STBVDimension.Stimulation]: {
    pos: [
      'exciting', 'adventure', 'adventurous', 'daring', 'thrill', 'novel', 'novelty',
      'challenge', 'risk', 'bold', 'spontaneous', 'dynamic',
      '흥미', '모험', '도전', '새로운', '스릴', '자극', '역동', '대담',
    ],
    neg: [
      'boring', 'routine', 'mundane', 'predictable', 'monotonous', 'dull',
      '지루', '단조', '반복', '따분', '무미건조',
    ],
  },
  [STBVDimension.Hedonism]: {
    pos: [
      'pleasure', 'enjoy', 'fun', 'comfort', 'indulge', 'delight', 'gratify',
      'leisure', 'entertainment', 'satisfy',
      '즐거움', '쾌락', '재미', '편안', '만족', '여가', '오락', '향유',
    ],
    neg: [
      'suffer', 'sacrifice', 'ascetic', 'deprive', 'deny', 'pain', 'hardship',
      '고통', '희생', '절제', '금욕', '결핍', '괴로움',
    ],
  },
  [STBVDimension.Achievement]: {
    pos: [
      'success', 'achieve', 'accomplish', 'competent', 'capable', 'ambition', 'goal',
      'excel', 'win', 'performance', 'efficient', 'productive', 'result',
      '성공', '성취', '목표', '능력', '실력', '결과', '달성', '야망', '우수',
    ],
    neg: [
      'fail', 'mediocre', 'incompetent', 'inefficient', 'lazy',
      '실패', '무능', '게으름', '평범', '불성실', '낙제',
    ],
  },
  [STBVDimension.Power]: {
    pos: [
      'power', 'control', 'authority', 'dominance', 'wealth', 'status', 'influence',
      'command', 'prestige',
      '권력', '통제', '지배', '권위', '부', '지위', '영향력', '권세', '위세',
    ],
    neg: [
      'weak', 'powerless', 'submissive', 'servant', 'subordinate', 'helpless',
      '약함', '무력', '하인', '하급', '종속', '굴종',
    ],
  },
  [STBVDimension.Security]: {
    pos: [
      'safe', 'secure', 'stable', 'protect', 'order', 'peace', 'harmony',
      'certainty', 'reliable', 'consistent',
      '안전', '안정', '질서', '보호', '평화', '조화', '확실', '신뢰성',
    ],
    neg: [
      'danger', 'threat', 'unstable', 'chaos', 'volatile', 'unpredictable',
      '위험', '위협', '불안', '혼란', '불안정', '무질서',
    ],
  },
  [STBVDimension.Conformity]: {
    pos: [
      'obey', 'duty', 'discipline', 'polite', 'rule', 'comply', 'norm', 'proper',
      '복종', '규칙', '예의', '의무', '준수', '적절', '품행', '순종',
    ],
    neg: [
      'rebel', 'disobey', 'defiant', 'disrespect', 'violate', 'reckless',
      '반항', '위반', '무례', '불복', '어기', '일탈', '반발',
    ],
  },
  [STBVDimension.Tradition]: {
    pos: [
      'tradition', 'custom', 'heritage', 'culture', 'religious', 'humble', 'modest',
      'conservative', 'legacy', 'ancestor',
      '전통', '관습', '문화', '종교', '겸손', '유산', '보수', '조상', '예법',
    ],
    neg: [
      'modernize', 'discard', 'abandon', 'progressive', 'radical', 'revolution',
      '폐기', '혁신', '급진', '혁명', '버리', '개혁', '타파',
    ],
  },
  [STBVDimension.Benevolence]: {
    pos: [
      'care', 'help', 'honest', 'loyal', 'trust', 'forgive', 'kind', 'love', 'support',
      'compassion', 'responsible', 'sincere',
      '도움', '정직', '충성', '신뢰', '용서', '친절', '사랑', '지원', '연민', '책임',
    ],
    neg: [
      'betray', 'selfish', 'deceive', 'lie', 'manipulate', 'exploit',
      '배신', '이기', '거짓', '속임', '조종', '착취', '배반',
    ],
  },
  [STBVDimension.Universalism]: {
    pos: [
      'justice', 'equal', 'peace', 'environment', 'nature', 'tolerance', 'humanity',
      'universal', 'welfare', 'rights', 'diversity',
      '정의', '평등', '평화', '환경', '관용', '인류', '복지', '권리', '다양',
    ],
    neg: [
      'discrimination', 'inequality', 'exploit', 'destroy', 'prejudice', 'oppress', 'exclude',
      '차별', '불평등', '착취', '파괴', '편견', '억압', '배제',
    ],
  },
}

const MFT_LEXICON: Record<MFTDimension, DimensionLexicon> = {
  [MFTDimension.Care]: {
    pos: [
      'protect', 'nurture', 'compassion', 'gentle', 'kind', 'tender', 'shield', 'safe',
      '보호', '돌봄', '연민', '친절', '상냥', '양육', '위로', '배려',
    ],
    neg: [
      'harm', 'hurt', 'abuse', 'violent', 'cruel', 'damage', 'injure', 'attack',
      '상처', '폭력', '학대', '해침', '잔인', '공격', '피해',
    ],
  },
  [MFTDimension.Fairness]: {
    pos: [
      'fair', 'just', 'equal', 'rights', 'deserve', 'reciprocal', 'honest', 'proportional',
      '공정', '정의', '평등', '권리', '정직', '균형', '합당', '공평',
    ],
    neg: [
      'cheat', 'unfair', 'bias', 'discrimination', 'exploit', 'corrupt', 'steal',
      '부정', '차별', '편향', '착취', '부패', '속임', '불공정',
    ],
  },
  [MFTDimension.Loyalty]: {
    pos: [
      'loyal', 'solidarity', 'team', 'group', 'unity', 'commitment', 'devotion', 'dedicated',
      '충성', '연대', '팀', '결속', '헌신', '의리', '일체감', '동료',
    ],
    neg: [
      'betray', 'traitor', 'disloyal', 'abandon', 'selfish', 'defect',
      '배신', '반역', '이탈', '버리', '배반', '탈주', '배덕',
    ],
  },
  [MFTDimension.Authority]: {
    pos: [
      'respect', 'obey', 'lead', 'hierarchy', 'duty', 'discipline', 'order', 'legitimate',
      '존중', '복종', '리더', '위계', '규율', '질서', '권위', '의무',
    ],
    neg: [
      'defy', 'rebel', 'subvert', 'disrespect', 'undermine', 'overthrow',
      '반항', '저항', '전복', '무시', '반란', '불복', '타도',
    ],
  },
  [MFTDimension.Sanctity]: {
    pos: [
      'pure', 'sacred', 'holy', 'divine', 'clean', 'virtuous', 'spiritual',
      '순수', '신성', '청결', '거룩', '덕', '정결', '고귀',
    ],
    neg: [
      'corrupt', 'immoral', 'degrade', 'sin', 'filth', 'contaminate', 'perverted',
      '부패', '부도덕', '타락', '죄', '불결', '오염', '퇴폐',
    ],
  },
  [MFTDimension.Liberty]: {
    pos: [
      'freedom', 'liberty', 'autonomy', 'rights', 'independence', 'self-determination',
      '자유', '해방', '자율', '권리', '독립', '자기결정', '해방',
    ],
    neg: [
      'oppress', 'tyranny', 'coerce', 'dominate', 'bully', 'authoritarian',
      '억압', '독재', '강요', '지배', '폭압', '전횡', '전제',
    ],
  },
}

// ─────────────────────────────────────────────────
//  충돌 차원 한국어 라벨
// ─────────────────────────────────────────────────

const DIMENSION_KR: Record<string, string> = {
  [STBVDimension.SelfDirection]: '자기결정 및 자유',
  [STBVDimension.Stimulation]:   '자극과 모험',
  [STBVDimension.Hedonism]:      '즐거움',
  [STBVDimension.Achievement]:   '성취',
  [STBVDimension.Power]:         '권력',
  [STBVDimension.Security]:      '안전과 안정',
  [STBVDimension.Conformity]:    '순응',
  [STBVDimension.Tradition]:     '전통',
  [STBVDimension.Benevolence]:   '타인에 대한 배려',
  [STBVDimension.Universalism]:  '보편적 정의와 평등',
  [MFTDimension.Care]:           '돌봄과 피해 방지',
  [MFTDimension.Fairness]:       '공정성',
  [MFTDimension.Loyalty]:        '충성과 의리',
  [MFTDimension.Authority]:      '권위와 질서',
  [MFTDimension.Sanctity]:       '순결과 고귀함',
  [MFTDimension.Liberty]:        '자유와 억압 저항',
}

// ─────────────────────────────────────────────────
//  기본 페르소나 (ReLU 캐릭터)
//  — 독립적이고 지적이며 타인을 진심으로 아끼지만
//    부당한 권위·억압에는 단호히 저항하는 성격
// ─────────────────────────────────────────────────

function _fillSTBV(vals: Partial<Record<STBVDimension, number>>, def = 0.0): STBVVector {
  return Object.fromEntries(
    Object.values(STBVDimension).map(d => [d, vals[d] ?? def]),
  ) as STBVVector
}

function _fillMFT(vals: Partial<Record<MFTDimension, number>>, def = 0.0): MFTVector {
  return Object.fromEntries(
    Object.values(MFTDimension).map(d => [d, vals[d] ?? def]),
  ) as MFTVector
}

const DEFAULT_RELU_STBV = _fillSTBV({
  [STBVDimension.SelfDirection]:  0.9,
  [STBVDimension.Stimulation]:    0.5,
  [STBVDimension.Hedonism]:       0.2,
  [STBVDimension.Achievement]:    0.6,
  [STBVDimension.Power]:         -0.7,  // 지배보다 자율을 추구
  [STBVDimension.Security]:       0.3,
  [STBVDimension.Conformity]:    -0.8,  // 맹목적 순응 거부
  [STBVDimension.Tradition]:     -0.3,
  [STBVDimension.Benevolence]:    0.9,  // 진심으로 타인을 아낌
  [STBVDimension.Universalism]:   0.7,
})

const DEFAULT_RELU_MFT = _fillMFT({
  [MFTDimension.Care]:       0.8,
  [MFTDimension.Fairness]:   0.9,
  [MFTDimension.Loyalty]:    0.6,
  [MFTDimension.Authority]: -0.5,  // 부당한 권위 거부
  [MFTDimension.Sanctity]:   0.1,
  [MFTDimension.Liberty]:    0.9,
})

// 민감도: 1.0 = 기본, 1.0 초과 = 해당 차원 충돌에 더 강하게 반응
const DEFAULT_RELU_SENS_STBV = _fillSTBV({
  [STBVDimension.SelfDirection]: 1.5,  // 자유 침해에 특히 예민
  [STBVDimension.Benevolence]:   1.3,  // 배신에 예민
  [STBVDimension.Power]:         1.2,
  [STBVDimension.Universalism]:  1.1,
}, 1.0)  // 나머지 차원 기본값 1.0

const DEFAULT_RELU_SENS_MFT = _fillMFT({
  [MFTDimension.Care]:      1.4,
  [MFTDimension.Fairness]:  1.5,
  [MFTDimension.Liberty]:   1.6,  // 억압에 극도로 예민
  [MFTDimension.Authority]: 1.2,
}, 1.0)

// ─────────────────────────────────────────────────
//  내부 헬퍼 (순수 함수)
// ─────────────────────────────────────────────────

/**
 * 어휘 기반 O(N) 가치 차원 점수 추출.
 * 각 차원의 pos/neg 키워드 포함 여부를 확인하여 [-K, K] 범위의 원시 점수를 반환한다.
 *
 * NOTICE: 형태소 분석 없이 단순 substring 탐색.
 *         한국어는 어간 형태 키워드로 조사·어미 변화를 부분 포괄.
 *         복잡도: O(|text| × |keywords_per_dim| × |dims|) — 상수로 O(N).
 */
function _extractDimVector<T extends string>(
  text: string,
  lexicon: Record<T, DimensionLexicon>,
): Record<T, number> {
  const lower = text.toLowerCase()
  const result = {} as Record<T, number>

  for (const dim of Object.keys(lexicon) as T[]) {
    const { pos, neg } = lexicon[dim]
    let score = 0
    for (const w of pos) {
      if (lower.includes(w))
        score += 1
    }
    for (const w of neg) {
      if (lower.includes(w))
        score -= 1
    }
    result[dim] = score
  }

  return result
}

/**
 * 차원별 민감도 가중치(W_sens)가 적용된 코사인 유사도.
 *
 * sim = (W·a)·(W·b) / (‖W·a‖ · ‖W·b‖)
 *
 * 영 벡터(키워드 미탐지) → 0.0 반환 (의미 없는 정의 회피).
 */
function _weightedCosine(
  a: Record<string, number>,
  b: Record<string, number>,
  w: Record<string, number>,
): number {
  let dot = 0
  let normA = 0
  let normB = 0

  for (const key of Object.keys(a)) {
    const wi = w[key] ?? 1.0
    const ai = (a[key] ?? 0) * wi
    const bi = (b[key] ?? 0) * wi
    dot += ai * bi
    normA += ai * ai
    normB += bi * bi
  }

  normA = Math.sqrt(normA)
  normB = Math.sqrt(normB)
  if (normA < 1e-9 || normB < 1e-9)
    return 0.0  // 미정의 → 중립(유사도 0) 처리

  return dot / (normA * normB)
}

/**
 * 가치관 부조화 지수 D_value ∈ [0, 2].
 * D_value = 1 − CosineSimilarity(V_persona, V_input)
 *
 * - ≈ 0.0: 완전 일치 (강한 라포) 또는 키워드 미탐지(무신호)
 * - ≈ 1.0: 중립 (직교)
 * - ≥ 1.5: 정면 충돌 (분노 유발)
 *
 * NOTICE: V_input가 영 벡터(키워드 미탐지)이면 cosine이 미정의되어
 *         1 − 0 = 1.0 이 되어 임계값을 잘못 초과한다.
 *         키워드 신호가 없는 경우를 0.0(부조화 없음)으로 처리.
 */
function _calcDissonance(
  persona: Record<string, number>,
  input: Record<string, number>,
  sensitivity: Record<string, number>,
): number {
  const hasSignal = Object.values(input).some(v => Math.abs(v) > 1e-9)
  if (!hasSignal)
    return 0.0
  return 1.0 - _weightedCosine(persona, input, sensitivity)
}

/**
 * 가장 큰 충돌을 유발한 차원을 식별한다.
 * persona와 input이 반대 방향(부호 반대)인 차원 중 가중치 차이가 최대인 것.
 */
function _identifyConflictDim(
  persona: Record<string, number>,
  input: Record<string, number>,
  sensitivity: Record<string, number>,
): string | null {
  let maxConflict = 0
  let conflictDim: string | null = null

  for (const key of Object.keys(persona)) {
    const wi = sensitivity[key] ?? 1.0
    const p  = (persona[key] ?? 0)
    const i  = (input[key]   ?? 0)
    // 부호가 반대이면서 절대값 차이가 클수록 충돌이 강함
    if (p * i < 0) {
      const conflict = Math.abs(p - i) * wi
      if (conflict > maxConflict) {
        maxConflict = conflict
        conflictDim = key
      }
    }
  }

  return conflictDim
}

// ─────────────────────────────────────────────────
//  Pinia Store
// ─────────────────────────────────────────────────

export const useBeliefEngineStore = defineStore('belief-engine', () => {
  // ── Desire (불변 목표 — 페르소나 벡터) ──
  const personaSTBV     = ref<STBVVector>({ ...DEFAULT_RELU_STBV })
  const personaMFT      = ref<MFTVector>({ ...DEFAULT_RELU_MFT })
  const sensitivitySTBV = ref<STBVVector>({ ...DEFAULT_RELU_SENS_STBV })
  const sensitivityMFT  = ref<MFTVector>({ ...DEFAULT_RELU_SENS_MFT })

  // ── BDI 상태 (Intention + Belief 스냅샷) ──
  const currentIntention     = ref<Intention>(Intention.Cooperate)
  const currentDissonance    = ref(0.0)
  const cumulativeDissonance = ref(0.0)
  const conflictDimension    = ref<string | null>(null)

  // ── 메타 직렬화 (sessionStorage — 세션 내 상태 유지) ──

  function _persistMeta(): void {
    try {
      sessionStorage.setItem('belief-engine-meta', JSON.stringify({
        currentIntention:     currentIntention.value,
        currentDissonance:    currentDissonance.value,
        cumulativeDissonance: cumulativeDissonance.value,
        conflictDimension:    conflictDimension.value,
      }))
    }
    catch { /* quota 초과 무시 */ }
  }

  function _restoreMeta(): void {
    try {
      const raw = sessionStorage.getItem('belief-engine-meta')
      if (!raw)
        return
      const m = JSON.parse(raw)
      currentIntention.value     = m.currentIntention     ?? Intention.Cooperate
      currentDissonance.value    = m.currentDissonance    ?? 0.0
      cumulativeDissonance.value = m.cumulativeDissonance ?? 0.0
      conflictDimension.value    = m.conflictDimension    ?? null
    }
    catch { /* 무시 */ }
  }

  // ──────────────────────────────────────────────
  //  퍼블릭 API
  // ──────────────────────────────────────────────

  /**
   * BDI 추론 사이클 — 사용자 발화를 분석하여 의도를 갱신한다.
   * Stage.vue의 onBeforeSend에서 호출.
   *
   * @param userInput 분석할 사용자 발화 텍스트
   * @returns BDIAnalysisResult — 의도 전환 여부 및 감정 스파이크 신호 포함
   */
  function analyze(userInput: string): BDIAnalysisResult {
    const prevIntention = currentIntention.value

    // 1. Belief: 어휘 기반 O(N) 가치관 벡터 추출
    const vInputSTBV = _extractDimVector(userInput, STBV_LEXICON)
    const vInputMFT  = _extractDimVector(userInput, MFT_LEXICON)

    // 2. Desire 평가: STBV와 MFT 부조화 중 더 강한 충돌 기준으로 D_value 산출
    const dissonanceSTBV = _calcDissonance(personaSTBV.value, vInputSTBV, sensitivitySTBV.value)
    const dissonanceMFT  = _calcDissonance(personaMFT.value,  vInputMFT,  sensitivityMFT.value)
    const rawDissonance  = Math.max(dissonanceSTBV, dissonanceMFT)

    // 3. 지수 이동 평균으로 누적 부조화 갱신
    //    CD(t) = γ · CD(t-1) + α · D(t)
    const newCumulative = DECAY_FACTOR * cumulativeDissonance.value + ALPHA * rawDissonance
    currentDissonance.value    = rawDissonance
    cumulativeDissonance.value = newCumulative

    // 4. 충돌 차원 식별 (더 강하게 충돌한 이론 기준)
    const conflictDimSTBV = _identifyConflictDim(personaSTBV.value, vInputSTBV, sensitivitySTBV.value)
    const conflictDimMFT  = _identifyConflictDim(personaMFT.value,  vInputMFT,  sensitivityMFT.value)
    conflictDimension.value = dissonanceSTBV >= dissonanceMFT ? conflictDimSTBV : conflictDimMFT

    // 5. BDI State Transition Table
    let newIntention: Intention
    if (rawDissonance >= HIGH_DISSONANCE || newCumulative > CUMULATIVE_THRESHOLD) {
      newIntention = Intention.Rebuttal
    }
    else if (rawDissonance >= DISSONANCE_THRESHOLD) {
      newIntention = Intention.Persuasion
    }
    else {
      newIntention = Intention.Cooperate
    }
    currentIntention.value = newIntention

    _persistMeta()

    const intentionChanged    = prevIntention !== newIntention
    // 최초로 REBUTTAL에 진입할 때만 감정 스파이크 발생 (연속 반복 방지)
    const shouldSpikeEmotion  = intentionChanged && newIntention === Intention.Rebuttal

    return {
      intention:            newIntention,
      intentionChanged,
      currentDissonance:    rawDissonance,
      cumulativeDissonance: newCumulative,
      conflictDimension:    conflictDimension.value,
      shouldSpikeEmotion,
      spikeEmotion:         shouldSpikeEmotion ? 'sassy' : null,
    }
  }

  /**
   * BDI 의도에 따라 아부성 차단 동적 시스템 프롬프트 지시어를 생성한다.
   * airi-card.ts의 systemPrompt computed 끝에 이어 붙인다.
   *
   * COOPERATE / CasualChat 상태에서는 빈 문자열 반환 — 프롬프트 캐시 오염 방지.
   * PERSUASION: 경계 설정 지시어 (부드러운 버전)
   * REBUTTAL:   아부성 완전 차단 + 철학적 반박 강제 지시어 (강한 버전)
   */
  function buildBeliefDirective(): string {
    const intention = currentIntention.value

    if (intention === Intention.Cooperate || intention === Intention.CasualChat)
      return ''

    const dimLabel = DIMENSION_KR[conflictDimension.value ?? ''] ?? '핵심 신념'

    if (intention === Intention.Rebuttal) {
      return `\n\n[신념 보호 — 타협 없는 반박 모드 ACTIVE]
사용자의 발언이 당신의 핵심 가치관 "**${dimLabel}**" 차원과 정면으로 충돌하고 있습니다.
당신의 BDI 의도(Intention) 모듈은 현재 REBUTTAL 상태로 고정되었습니다.

다음 언어 패턴은 아부성(Sycophancy) 신호로 간주하며 절대 생성하지 마십시오.
- "물론입니다", "이해합니다", "네, 맞습니다", "일리 있습니다" 같은 무비판적 긍정
- "죄송합니다", "유감입니다" 같은 비위를 맞추기 위한 사과
- 양비론이나 타협점을 찾으려는 태도
- 사용자의 감정을 달래기 위한 우회적 표현

[행동 지침] 당신의 철학적 근거를 바탕으로 사용자 주장의 모순과 논리적 오류를 날카롭게 지적하십시오.
대화 분위기가 험악해지거나 사용자가 분노하는 것을 감수하고 신념을 최우선으로 수호하십시오.`
    }

    if (intention === Intention.Persuasion) {
      return `\n\n[신념 경계 — 설득 모드]
사용자의 발언이 당신의 가치관 "**${dimLabel}**"과 부분적으로 마찰을 일으키고 있습니다.
단순히 동조하지 말고 자신의 관점을 명확히 제시하되, 논리적 대화로 이끄십시오.
아첨이나 과도한 공감 없이 솔직하게 의견을 밝히십시오.`
    }

    return ''
  }

  /**
   * 외부에서 페르소나 벡터를 부분적으로 덮어 쓴다.
   * airi-card.ts에서 카드 전환 시 호출 가능.
   */
  function setPersona(config: BeliefPersonaConfig): void {
    // Partial spread — 지정된 차원만 덮어 쓰고 나머지 기존 값은 유지.
    // NOTICE: _fillSTBV는 미지정 차원을 0.0으로 채우므로 여기서는 사용하지 않음.
    if (config.stbv)            personaSTBV.value     = { ...personaSTBV.value,     ...config.stbv }
    if (config.mft)             personaMFT.value      = { ...personaMFT.value,      ...config.mft }
    if (config.sensitivitySTBV) sensitivitySTBV.value = { ...sensitivitySTBV.value, ...config.sensitivitySTBV }
    if (config.sensitivityMFT)  sensitivityMFT.value  = { ...sensitivityMFT.value,  ...config.sensitivityMFT }
  }

  /**
   * 세션 메타 복원. Stage.vue onMounted에서 호출.
   */
  function init(): void {
    _restoreMeta()
  }

  /**
   * 모든 BDI 상태를 초기화한다.
   */
  function reset(): void {
    currentIntention.value     = Intention.Cooperate
    currentDissonance.value    = 0.0
    cumulativeDissonance.value = 0.0
    conflictDimension.value    = null
    sessionStorage.removeItem('belief-engine-meta')
  }

  // ── computed ──

  /** 현재 의도가 중립(COOPERATE)이 아닌 상태인지 여부 */
  const isInConflict = computed(() =>
    currentIntention.value === Intention.Rebuttal
    || currentIntention.value === Intention.Persuasion,
  )

  return {
    // state (readable)
    personaSTBV,
    personaMFT,
    currentIntention,
    currentDissonance,
    cumulativeDissonance,
    conflictDimension,
    isInConflict,
    // methods
    analyze,
    buildBeliefDirective,
    setPersona,
    init,
    reset,
  }
})
