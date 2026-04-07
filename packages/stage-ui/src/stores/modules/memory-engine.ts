/**
 * Memory Engine v1 — 차별 기억 엔진 (감정-기억 통합 인지 아키텍처)
 * ================================================================
 * emotion-engine.ts 와 연동하여 사용하는 독립 실행 가능 기억 엔진.
 *
 * [이론 기반]
 * 1. ACT-R 기저 활성화 (Anderson, 1983)         — 멱법칙 망각 + 간격 반복 내재화
 * 2. eCMR 코사인 유사도 (Polyn et al., 2009)    — 연속형 감정 벡터 기반 기분 일치 인출
 * 3. Yerkes-Dodson 법칙 (1908)                   — 에너지/부조화 조합별 인코딩 품질 분기
 * 4. Flashbulb Memory (Brown & Kulik, 1977)      — 고각성+풍부한 에너지 순간 망각 면역
 * 5. 주기적 GC 수면 사이클                       — 비활성 기억 제거 (LLM 호출 없음)
 *
 * [과잉 설계 항목 간소화]
 * - A-MAC 5차원 → importance = max(emotion_dist) * (1 + dissonance * 0.5)
 * - ACAN 크로스 어텐션 → eCMR 코사인 유사도로 흡수
 * - 수면 사이클 LLM 병합 → GC만 (v2에서 의미 기억 병합 추가)
 *
 * [감정 엔진 연동 포인트]
 * - EmotionEngineResult.probabilities → emotionDist (6-dim vector)
 * - EmotionEngineResult.energy        → 인코딩 품질 게이트
 * - EmotionEngineResult.dissonance    → importance 부스트 + flashbulb 트리거
 * - EmotionEngineResult.state         → dominantEmotion 라벨
 */

import { nanoid } from 'nanoid'
import { defineStore } from 'pinia'
import { ref } from 'vue'

import { useDatabaseStore } from '../database'

// ─────────────────────────────────────────────────
//  감정 한국어 라벨 (emotion-engine.ts와 동기화)
// ─────────────────────────────────────────────────

const EMOTION_KR: Record<string, string> = {
  happy: '기분좋음',
  neutral: '평온함',
  sassy: '깐찐함',
  tired: '귀찮음',
  excited: '설렘',
  confused: '혼란',
}

// ─────────────────────────────────────────────────
//  하이퍼파라미터
// ─────────────────────────────────────────────────

const DECAY_BASE = 0.5       // ACT-R 기본 감쇠 계수 d
const DECAY_FLASHBULB = 0.05 // flashbulb/semantic — 사실상 영구 보존
const ECMR_GAMMA = 0.5       // eCMR 감정 일치도 스케일 γ
const ACTIVATION_THRESHOLD = -3.0  // GC 임계값 (B_i 이하 기억 삭제)
const SLEEP_CYCLE_INTERVAL = 20    // 수면 사이클 발동 간격 (턴)
const MAX_MEMORIES = 200           // 최대 기억 보유 수

// Yerkes-Dodson 임계값
const YD_HIGH_DISSONANCE = 0.8
const YD_LOW_ENERGY = 0.2
const YD_LOW_DISSONANCE = 0.3
const YD_EXHAUSTED_ENERGY = 0.1

// ─────────────────────────────────────────────────
//  타입 정의
// ─────────────────────────────────────────────────

export enum MemoryType {
  Episodic = 'episodic', // 단기적 경험/일화
  Semantic = 'semantic', // 압축된 장기 지식 (v2 수면 사이클에서 생성)
}

/**
 * 기억 저장 모드 — 어떤 내용을 기억할지 선택.
 * - AssistantOnly: Airi가 한 말만 기억 (구 기본값)
 * - UserOnly:      사용자가 한 말만 기억
 * - Exchange:      한 턴 전체 저장 ("User: X\nAiri: Y") — 맥락 보존에 최적 (기본값)
 * - Disabled:      기억 저장 비활성화
 */
export enum MemoryStorageMode {
  AssistantOnly = 'assistant-only',
  UserOnly = 'user-only',
  Exchange = 'exchange',
  Disabled = 'disabled',
}

export enum EncodingQuality {
  Full = 'full',             // 완전 인코딩 (LTP 최적 조건)
  Fragmented = 'fragmented', // 파편화 인코딩 (고부조화 + 에너지 고갈)
  Degraded = 'degraded',     // 저품질 인코딩 (일상적 소진)
}

export interface MemoryEntry {
  memoryId: string          // nanoid — 고유 식별자
  content: string           // 기억 내용 (파편화 시 잘릴 수 있음)
  memoryType: MemoryType    // Episodic / Semantic
  emotionDist: number[]     // 저장 당시 6-dim 감정 확률 분포 (eCMR용)
  dominantEmotion: string   // EmotionEngineResult.state 값 (순환 의존 방지 위해 string)
  energy: number            // 저장 당시 에너지 (인코딩 품질 결정)
  dissonance: number        // 저장 당시 부조화 (중요도 부스트)
  importance: number        // 0.0 ~ 1.0 (A-MAC 간소화 스코어)
  createdTurn: number       // 생성 회차 (conversationTurnCount 기준)
  accessHistory: number[]   // 인출 시점 이력 — ACT-R Σ 연산용
  flashbulb: boolean        // 망각 면역 플래그
  encodingQuality: EncodingQuality
}

export interface RetrievalResult {
  memory: MemoryEntry
  score: number      // 최종 인출 스코어 (B_i + eCMR 가중치)
  activation: number // ACT-R 기저 활성화 B_i
  moodMatch: number  // eCMR 코사인 유사도 기반 감정 일치도
}

export interface MemoryEngineSerializable {
  currentTurn: number
  turnsSinceSleep: number
  memories: MemoryEntry[]  // 하위 호환용 — 기본 저장은 DuckDB
  storageMode: MemoryStorageMode
}

// ─────────────────────────────────────────────────
//  내부 헬퍼 (순수 함수)
// ─────────────────────────────────────────────────

/**
 * 코사인 유사도 — eCMR 감정 벡터 간 유사도 측정.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  normA = Math.sqrt(normA)
  normB = Math.sqrt(normB)
  if (normA < 1e-12 || normB < 1e-12)
    return 0
  return dot / (normA * normB)
}

/**
 * ACT-R 감쇠 계수 d_i.
 * - flashbulb / Semantic → 0.05 (영구 보존에 가까움)
 * - 일반 기억            → 0.5 * (1 - importance) (중요할수록 느리게 감쇠)
 */
function decayParam(m: MemoryEntry): number {
  if (m.flashbulb || m.memoryType === MemoryType.Semantic)
    return DECAY_FLASHBULB
  return Math.max(0.05, DECAY_BASE * (1.0 - m.importance))
}

/**
 * ACT-R 기저 활성화 수준 B_i.
 *
 * B_i = ln( Σ_{t_k ∈ accessHistory} (currentTurn - t_k + 1)^(-d) )
 *
 * - 멱법칙 망각: 지수 감쇠보다 오래된 기억이 더 잘 살아남음
 * - 간격 반복 내재화: 자주 인출될수록 Σ 항이 누적 → B_i↑
 */
function baseActivation(m: MemoryEntry, currentTurn: number): number {
  const d = decayParam(m)
  const history = m.accessHistory.length > 0 ? m.accessHistory : [m.createdTurn]
  let total = 0
  for (const t of history) {
    const delta = currentTurn - t + 1
    if (delta > 0)
      total += Math.pow(delta, -d)
  }
  if (total <= 0)
    return Number.NEGATIVE_INFINITY
  return Math.log(total)
}

/**
 * 최종 인출 스코어 = B_i + γ * cosine_similarity(enc_dist, cur_dist)
 */
function retrievalScore(
  m: MemoryEntry,
  currentDist: number[],
  currentTurn: number,
): [score: number, activation: number, moodMatch: number] {
  const activation = baseActivation(m, currentTurn)
  const cosine = cosineSimilarity(m.emotionDist, currentDist)
  const moodMatch = 1.0 + ECMR_GAMMA * cosine
  const score = activation + ECMR_GAMMA * cosine
  return [score, activation, moodMatch]
}

/**
 * Yerkes-Dodson 역U자 법칙 기반 인코딩 품질.
 *
 * 고부조화 + 충분한 에너지  → Full       (편도체 LTP 최적)
 * 고부조화 + 에너지 고갈    → Fragmented (인지 과부하, 파편화)
 * 일상적 소진               → Degraded   (저해상도, 빠른 망각)
 * 그 외                     → Full
 */
function calcEncodingQuality(energy: number, dissonance: number): EncodingQuality {
  if (dissonance > YD_HIGH_DISSONANCE && energy > 0.5)
    return EncodingQuality.Full
  if (dissonance > YD_HIGH_DISSONANCE && energy < YD_LOW_ENERGY)
    return EncodingQuality.Fragmented
  if (dissonance < YD_LOW_DISSONANCE && energy < YD_EXHAUSTED_ENERGY)
    return EncodingQuality.Degraded
  return EncodingQuality.Full
}

/**
 * Flashbulb 조건 (Yerkes-Dodson 역U자 정점):
 * 높은 각성(dissonance↑) + 충분한 인지 자원(energy↑) → 망각 면역.
 * NOTICE: energy < 0.1 → flashbulb 는 Yerkes-Dodson 위반이므로 사용하지 않음.
 *         에너지 고갈 상태는 오히려 Fragmented 인코딩으로 분기.
 */
function isFlashbulb(energy: number, dissonance: number): boolean {
  return dissonance > YD_HIGH_DISSONANCE && energy > 0.5
}

/**
 * A-MAC 간소화 중요도.
 * importance = max(emotionDist) * (1 + dissonance * 0.5)
 * LLM 호출 없이 감정 엔진 상태만으로 즉시 산출.
 */
function calcImportance(emotionDist: number[], dissonance: number): number {
  const peak = emotionDist.length > 0 ? Math.max(...emotionDist) : 0
  return Math.min(1.0, peak * (1.0 + 0.5 * dissonance))
}

/**
 * 파편화 인코딩: 앞 절반 단어만 보존 + 맥락 손실 표시.
 * 에너지 고갈 + 고부조화 → 불완전한 기억 형성 시뮬레이션.
 */
function fragmentContent(content: string): string {
  const words = content.split(' ')
  const kept = words.slice(0, Math.max(3, Math.floor(words.length / 2)))
  return `${kept.join(' ')}... (맥락 손실)`
}

// ─────────────────────────────────────────────────
//  기억 엔진 Pinia Store
// ─────────────────────────────────────────────────

export const useMemoryEngineStore = defineStore('memory-engine', () => {
  const dbStore = useDatabaseStore()

  const memories = ref<MemoryEntry[]>([])
  const currentTurn = ref(0)
  const turnsSinceSleep = ref(0)
  // Exchange: 한 턴(User + Airi) 전체를 하나의 기억으로 저장 — eCMR 맥락 보존에 최적
  const storageMode = ref<MemoryStorageMode>(MemoryStorageMode.Exchange)

  // ─────────────────────────────────────────────
  //  DuckDB 내부 헬퍼
  // ─────────────────────────────────────────────

  /**
   * SQL 문자열 내 단일 따옴표를 이중화하는 이스케이프 헬퍼.
   * NOTICE: DuckDB WASM execute()가 파라미터화 쿼리를 지원하지 않아 수동 이스케이프.
   *         모든 사용자 입력 문자열은 이 함수를 통과해야 함.
   */
  function _sqlEscape(s: string): string {
    return s.replace(/'/g, "''")
  }

  /**
   * MemoryEntry를 DuckDB에 INSERT/UPSERT.
   * 동기 store() 호출에서 fire-and-forget으로 사용 — await 없이 호출.
   */
  async function _dbUpsertMemory(entry: MemoryEntry): Promise<void> {
    if (!dbStore.isReady)
      return
    const data = _sqlEscape(JSON.stringify(entry))
    await dbStore.execute(
      `INSERT INTO memories (memory_id, data) VALUES ('${entry.memoryId}', '${data}')
       ON CONFLICT (memory_id) DO UPDATE SET data = excluded.data`,
    )
    _persistMeta()
  }

  /**
   * DuckDB에서 기억 삭제.
   */
  async function _dbDeleteMemories(ids: string[]): Promise<void> {
    if (!dbStore.isReady || ids.length === 0)
      return
    const idList = ids.map(id => `'${id}'`).join(', ')
    await dbStore.execute(`DELETE FROM memories WHERE memory_id IN (${idList})`)
    _persistMeta()
  }

  /**
   * retrieve() 후 변경된 access_history를 DuckDB에 반영.
   */
  async function _dbUpdateAccessHistory(entries: MemoryEntry[]): Promise<void> {
    if (!dbStore.isReady || entries.length === 0)
      return
    for (const entry of entries) {
      const data = _sqlEscape(JSON.stringify(entry))
      await dbStore.execute(
        `UPDATE memories SET data = '${data}' WHERE memory_id = '${entry.memoryId}'`,
      )
    }
  }

  // ── 메타 직렬화 (sessionStorage: 경량 상태만) ──

  interface MemoryEngineMeta {
    currentTurn: number
    turnsSinceSleep: number
    storageMode: MemoryStorageMode
  }

  function _persistMeta(): void {
    try {
      const meta: MemoryEngineMeta = {
        currentTurn: currentTurn.value,
        turnsSinceSleep: turnsSinceSleep.value,
        storageMode: storageMode.value,
      }
      sessionStorage.setItem('memory-engine-meta', JSON.stringify(meta))
    }
    catch {
      // quota 초과 시 무시
    }
  }

  function _restoreMeta(): void {
    try {
      const raw = sessionStorage.getItem('memory-engine-meta')
      if (!raw)
        return
      const meta = JSON.parse(raw) as MemoryEngineMeta
      currentTurn.value = meta.currentTurn ?? 0
      turnsSinceSleep.value = meta.turnsSinceSleep ?? 0
      storageMode.value = meta.storageMode ?? MemoryStorageMode.Exchange
    }
    catch {
      // 무시
    }
  }

  // ── 내부 GC ──

  function _sleepCycleInternal(): number {
    const before = memories.value.length
    const removed: string[] = []

    memories.value = memories.value.filter((m) => {
      const keep = m.flashbulb
        || m.memoryType === MemoryType.Semantic
        || baseActivation(m, currentTurn.value) > ACTIVATION_THRESHOLD
      if (!keep)
        removed.push(m.memoryId)
      return keep
    })

    turnsSinceSleep.value = 0

    // fire-and-forget DB 삭제
    if (removed.length > 0)
      _dbDeleteMemories(removed)

    return before - memories.value.length
  }

  function _evictOne(): void {
    const candidates = memories.value.filter(m => !m.flashbulb)
    if (candidates.length === 0)
      return
    const worst = candidates.reduce(
      (a, b) => baseActivation(a, currentTurn.value) <= baseActivation(b, currentTurn.value) ? a : b,
    )
    memories.value = memories.value.filter(m => m.memoryId !== worst.memoryId)
    // fire-and-forget DB 삭제
    _dbDeleteMemories([worst.memoryId])
  }

  // ─────────────────────────────────────────────
  //  퍼블릭 API
  // ─────────────────────────────────────────────

  /**
   * 턴 증가 + 주기적 수면 사이클 트리거.
   * EmotionEngine의 conversationTurnCount 와 동기화해서 호출.
   */
  function tick(turn?: number): void {
    if (turn !== undefined)
      currentTurn.value = turn
    else
      currentTurn.value++

    turnsSinceSleep.value++
    if (turnsSinceSleep.value >= SLEEP_CYCLE_INTERVAL)
      _sleepCycleInternal()

    _persistMeta()
  }

  /**
   * 기억을 저장한다.
   *
   * @param content          기억할 텍스트 (대화 내용, 사건 요약 등)
   * @param emotionDist      EmotionEngineResult.probabilities 값 배열 (6-dim)
   * @param dominantEmotion  EmotionEngineResult.state
   * @param energy           EmotionEngineResult.energy
   * @param dissonance       EmotionEngineResult.dissonance
   * @param memoryType       기본값: Episodic
   */
  function store(params: {
    content: string
    emotionDist: number[]
    dominantEmotion: string
    energy: number
    dissonance: number
    memoryType?: MemoryType
  }): MemoryEntry {
    const { content, emotionDist, dominantEmotion, energy, dissonance, memoryType = MemoryType.Episodic } = params

    const quality = calcEncodingQuality(energy, dissonance)
    const flashbulb = isFlashbulb(energy, dissonance)

    // 파편화 인코딩: 내용 손상 시뮬레이션
    const storedContent = quality === EncodingQuality.Fragmented
      ? fragmentContent(content)
      : content

    let importance = calcImportance(emotionDist, dissonance)
    // Degraded 인코딩 → 중요도 강제 하향 (빠른 망각 유도)
    if (quality === EncodingQuality.Degraded)
      importance *= 0.4

    const entry: MemoryEntry = {
      memoryId: nanoid(),
      content: storedContent,
      memoryType,
      emotionDist: [...emotionDist],
      dominantEmotion,
      energy,
      dissonance,
      importance,
      createdTurn: currentTurn.value,
      accessHistory: [currentTurn.value],
      flashbulb,
      encodingQuality: quality,
    }

    memories.value.push(entry)

    // 최대 기억 수 초과 → 최저 활성화 기억 제거 (flashbulb 제외)
    if (memories.value.length > MAX_MEMORIES)
      _evictOne()

    // fire-and-forget DuckDB 저장 (동기 API를 유지하면서 영구 보존)
    _dbUpsertMemory(entry)
    return entry
  }

  /**
   * 현재 감정 벡터 기준으로 가장 관련성 높은 기억 topK개 반환.
   * 인출 시 accessHistory에 현재 턴 추가 → 간격 반복 효과 자동 반영.
   */
  function retrieve(emotionDist: number[], topK = 3): RetrievalResult[] {
    if (memories.value.length === 0)
      return []

    const scored: RetrievalResult[] = memories.value.map((m) => {
      const [score, activation, moodMatch] = retrievalScore(m, emotionDist, currentTurn.value)
      return { memory: m, score, activation, moodMatch }
    })

    scored.sort((a, b) => b.score - a.score)
    const top = scored.slice(0, topK)

    // 인출 이력 갱신 (다음 B_i 계산에 반영)
    for (const r of top)
      r.memory.accessHistory.push(currentTurn.value)

    // fire-and-forget DB 갱신
    _dbUpdateAccessHistory(top.map(r => r.memory))

    return top
  }

  /**
   * 인출된 기억을 LLM 시스템 프롬프트 삽입용 텍스트로 변환.
   * EmotionEngine.buildEmotionPrompt() 뒤에 이어 붙이면 된다.
   */
  function buildMemoryPrompt(emotionDist: number[], topK = 3): string {
    const results = retrieve(emotionDist, topK)
    if (results.length === 0)
      return ''

    const lines = ['\n\n[기억]']
    for (const r of results) {
      const m = r.memory
      const flag = m.flashbulb ? ' ★' : ''
      const qualityTag = m.encodingQuality === EncodingQuality.Fragmented
        ? ' [불완전]'
        : m.encodingQuality === EncodingQuality.Degraded
          ? ' [희미]'
          : ''
      const emotionLabel = EMOTION_KR[m.dominantEmotion] ?? m.dominantEmotion
      lines.push(`- (${emotionLabel}${flag}${qualityTag}) ${m.content}`)
    }
    return lines.join('\n')
  }

  /**
   * 수동으로 수면 사이클(GC) 트리거. 제거된 기억 수 반환.
   */
  function sleepCycle(): number {
    return _sleepCycleInternal()
  }

  // ── 초기화 / 복원 ──

  /**
   * DuckDB에서 모든 기억을 로드하고 세션 메타를 복원한다.
   * Stage.vue의 onMounted에서 DuckDB 테이블 생성 후 반드시 호출.
   */
  async function initFromDatabase(): Promise<void> {
    // 1. 세션 메타 복원 (current_turn, storage_mode 등)
    _restoreMeta()

    // 2. DuckDB에서 기억 로드
    if (!dbStore.isReady)
      return
    const rows = await dbStore.execute('SELECT data FROM memories')
    memories.value = rows.map((row) => {
      return JSON.parse(row.data as string) as MemoryEntry
    })
  }

  function setStorageMode(mode: MemoryStorageMode): void {
    storageMode.value = mode
    _persistMeta()
  }

  /**
   * 모든 기억을 초기화하고 DuckDB 테이블도 비운다.
   */
  async function reset(): Promise<void> {
    memories.value = []
    currentTurn.value = 0
    turnsSinceSleep.value = 0
    storageMode.value = MemoryStorageMode.Exchange
    sessionStorage.removeItem('memory-engine-meta')
    await dbStore.execute('DELETE FROM memories')
  }

  // ── 직렬화 인터페이스 (하위 호환 유지) ──

  function serialize(): MemoryEngineSerializable {
    return {
      currentTurn: currentTurn.value,
      turnsSinceSleep: turnsSinceSleep.value,
      memories: memories.value.map(m => ({ ...m })),
      storageMode: storageMode.value,
    }
  }

  function restore(data: MemoryEngineSerializable): void {
    currentTurn.value = data.currentTurn
    turnsSinceSleep.value = data.turnsSinceSleep ?? 0
    memories.value = data.memories.map(m => ({ ...m }))
    storageMode.value = data.storageMode ?? MemoryStorageMode.Exchange
  }

  return {
    memories,
    currentTurn,
    storageMode,
    tick,
    store,
    setStorageMode,
    retrieve,
    buildMemoryPrompt,
    sleepCycle,
    initFromDatabase,
    serialize,
    restore,
    reset,
  }
})
