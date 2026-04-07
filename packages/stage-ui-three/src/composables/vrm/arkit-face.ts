/**
 * ARKit 52 Blendshape Face Controller
 * =====================================
 * TypeScript port of face_arkit52.py
 *
 * EmotionEngineV2 → ARKit 52 blendshape 값으로 변환.
 * VTuber 표준 퍼펙트싱크(Perfect Sync) 형식을 완전 지원하며,
 * VRMExpressionManager를 통해 three-vrm에 직접 적용.
 *
 * 출력 형식:
 *   { blendshapes: Record<string, number> (52개), transition, micro_expressions, blink, viseme_sequence }
 */

import type { VRMCore } from '@pixiv/three-vrm-core'

import { ref } from 'vue'

// 감정 엔진 결과 인터페이스 (emotion-engine.ts의 EmotionEngineResult와 호환)
export interface ARKitEmotionInput {
  state: string
  probabilities: Record<string, number>
  prevState: string
  energy: number
  dissonance: number
  filler: string
}

// ─────────────────────────────────────────────────
//  ARKit 52 Blendshape 키 정의
// ─────────────────────────────────────────────────

export const ARKIT_KEYS = [
  // Eye (14)
  'eyeBlinkLeft', 'eyeBlinkRight',
  'eyeLookDownLeft', 'eyeLookDownRight',
  'eyeLookInLeft', 'eyeLookInRight',
  'eyeLookOutLeft', 'eyeLookOutRight',
  'eyeLookUpLeft', 'eyeLookUpRight',
  'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight',
  // Brow (5)
  'browDownLeft', 'browDownRight',
  'browInnerUp',
  'browOuterUpLeft', 'browOuterUpRight',
  // Jaw (4)
  'jawForward', 'jawLeft',
  'jawRight', 'jawOpen',
  // Mouth (22)
  'mouthClose',
  'mouthFunnel', 'mouthPucker',
  'mouthLeft', 'mouthRight',
  'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight',
  'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthStretchLeft', 'mouthStretchRight',
  'mouthRollLower', 'mouthRollUpper',
  'mouthShrugLower', 'mouthShrugUpper',
  'mouthPressLeft', 'mouthPressRight',
  'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight',
  // Nose (2)
  'noseSneerLeft', 'noseSneerRight',
  // Cheek (3)
  'cheekPuff',
  'cheekSquintLeft', 'cheekSquintRight',
  // Tongue (1)
  'tongueOut',
] as const

export type ARKitKey = typeof ARKIT_KEYS[number]
export type ARKitBlendshapes = Record<ARKitKey, number>

function zeros(): ARKitBlendshapes {
  return Object.fromEntries(ARKIT_KEYS.map(k => [k, 0.0])) as ARKitBlendshapes
}

// ─────────────────────────────────────────────────
//  감정별 기본 블렌드쉐이프 (Base Expressions)
//
//  각 감정이 100%일 때의 "완전한" 표정.
//  확률 분포에 따라 가중 블렌딩이 중간 표정을 생성.
//  좌우 비대칭을 약간 넣어야 자연스러움.
// ─────────────────────────────────────────────────

const BASE_EXPRESSIONS: Record<string, Partial<ARKitBlendshapes>> = {
  // HAPPY: 밝은 미소, 눈웃음
  happy: {
    eyeSquintLeft: 0.4, eyeSquintRight: 0.35,
    cheekSquintLeft: 0.35, cheekSquintRight: 0.3,
    mouthSmileLeft: 0.7, mouthSmileRight: 0.65,
    mouthDimpleLeft: 0.2, mouthDimpleRight: 0.2,
    browInnerUp: 0.15,
    browOuterUpLeft: 0.1, browOuterUpRight: 0.1,
    jawOpen: 0.08,
    mouthUpperUpLeft: 0.1, mouthUpperUpRight: 0.1,
  },
  // NEUTRAL: 무표정, 이완
  neutral: {
    mouthClose: 0.1,
    mouthPressLeft: 0.05, mouthPressRight: 0.05,
  },
  // SASSY: 한쪽 눈썹 올림, 삐딱한 입
  sassy: {
    browDownLeft: 0.35,
    browOuterUpRight: 0.5,        // 오른쪽 눈썹 올림 — 비대칭
    eyeSquintLeft: 0.3,
    eyeSquintRight: 0.1,
    mouthFrownLeft: 0.3,
    mouthFrownRight: 0.15,        // 비대칭 삐꼬리
    mouthLeft: 0.15,
    mouthPressLeft: 0.2, mouthPressRight: 0.1,
    noseSneerLeft: 0.2, noseSneerRight: 0.1,
    jawLeft: 0.05,
  },
  // TIRED: 처진 눈꺼풀, 힘없는 표정
  tired: {
    eyeBlinkLeft: 0.45, eyeBlinkRight: 0.5,
    eyeLookDownLeft: 0.3, eyeLookDownRight: 0.3,
    browDownLeft: 0.15, browDownRight: 0.15,
    browInnerUp: 0.2,             // 안쪽 눈썹 이지 올라감 (피곤한 느낌)
    jawOpen: 0.15,
    mouthFrownLeft: 0.15, mouthFrownRight: 0.15,
    mouthStretchLeft: 0.1, mouthStretchRight: 0.1,
    mouthRollLower: 0.1,
    cheekPuff: 0.05,
  },
  // EXCITED: 크게 뜬 눈, 환한 미소, 입 벌림
  excited: {
    eyeWideLeft: 0.6, eyeWideRight: 0.55,
    browInnerUp: 0.5,
    browOuterUpLeft: 0.55, browOuterUpRight: 0.5,
    mouthSmileLeft: 0.85, mouthSmileRight: 0.8,
    jawOpen: 0.35,
    mouthDimpleLeft: 0.3, mouthDimpleRight: 0.3,
    mouthUpperUpLeft: 0.2, mouthUpperUpRight: 0.2,
    mouthLowerDownLeft: 0.15, mouthLowerDownRight: 0.15,
    cheekSquintLeft: 0.5, cheekSquintRight: 0.45,
  },
  // CONFUSED: 갸우뚱, 비대칭 눈썹, 입 오므림
  confused: {
    browInnerUp: 0.4,
    browOuterUpLeft: 0.35,
    browOuterUpRight: 0.1,        // 비대칭 — 갸우뚱 느낌
    browDownRight: 0.15,
    eyeSquintRight: 0.15,
    eyeWideLeft: 0.2,
    jawOpen: 0.1,
    mouthFrownLeft: 0.1,
    mouthPucker: 0.15,
    mouthLeft: 0.1,
    mouthShrugLower: 0.2, mouthShrugUpper: 0.15,
    jawRight: 0.05,
  },
}

// ─────────────────────────────────────────────────
//  추임새 → 한글 자모 비지음(Viseme) 매핑
// ─────────────────────────────────────────────────

const VISEME_MAP: Record<string, Partial<ARKitBlendshapes>> = {
  // ── 모음 ──
  'ㅏ': { jawOpen: 0.55, mouthLowerDownLeft: 0.2, mouthLowerDownRight: 0.2 },
  'ㅐ': { jawOpen: 0.4, mouthFunnel: 0.15, mouthLowerDownLeft: 0.15, mouthLowerDownRight: 0.15 },
  'ㅗ': { jawOpen: 0.25, mouthFunnel: 0.4, mouthPucker: 0.3 },
  'ㅜ': { jawOpen: 0.15, mouthFunnel: 0.5, mouthPucker: 0.45 },
  'ㅡ': { jawOpen: 0.08, mouthStretchLeft: 0.25, mouthStretchRight: 0.25 },
  'ㅣ': { jawOpen: 0.1, mouthSmileLeft: 0.3, mouthSmileRight: 0.3, mouthStretchLeft: 0.2, mouthStretchRight: 0.2 },
  'ㅔ': { jawOpen: 0.35, mouthStretchLeft: 0.15, mouthStretchRight: 0.15 },
  'ㅓ': { jawOpen: 0.3, mouthStretchLeft: 0.2, mouthStretchRight: 0.2 },
  'ㅑ': { jawOpen: 0.5, mouthLowerDownLeft: 0.25, mouthLowerDownRight: 0.25, mouthUpperUpLeft: 0.1, mouthUpperUpRight: 0.1 },
  // ── 자음 (입 모양에 영향 주는 것만) ──
  'ㅁ': { mouthClose: 0.6, mouthPressLeft: 0.3, mouthPressRight: 0.3 },
  'ㅂ': { mouthClose: 0.5, mouthPressLeft: 0.25, mouthPressRight: 0.25 },
  'ㅎ': { jawOpen: 0.3, mouthSmileLeft: 0.2, mouthSmileRight: 0.2 },
  'ㅋ': { jawOpen: 0.2, mouthSmileLeft: 0.35, mouthSmileRight: 0.35, cheekSquintLeft: 0.15, cheekSquintRight: 0.15 },
  'ㅉ': { mouthPressLeft: 0.3, mouthPressRight: 0.3, mouthStretchLeft: 0.1, mouthStretchRight: 0.1 },
  'ㄷ': { jawOpen: 0.1, mouthClose: 0.15 },
  // ── 특수 문자 ──
  '.': { mouthClose: 0.3, mouthPressLeft: 0.15, mouthPressRight: 0.15 },
  '!': { jawOpen: 0.6, mouthSmileLeft: 0.2, mouthSmileRight: 0.2, mouthLowerDownLeft: 0.2, mouthLowerDownRight: 0.2 },
  '?': { jawOpen: 0.2, mouthFunnel: 0.1, mouthShrugLower: 0.15 },
  '~': { mouthSmileLeft: 0.25, mouthSmileRight: 0.25 },
}

// ─────────────────────────────────────────────────
//  한글 유니코드 → 자모 분해
// ─────────────────────────────────────────────────

const CHOSEONG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
const JUNGSEONG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
const JONGSEONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

function decomposeHangul(char: string): string[] {
  const code = char.charCodeAt(0) - 0xAC00
  if (code < 0 || code > 11171)
    return [char]
  const cho = Math.floor(code / (21 * 28))
  const jung = Math.floor((code % (21 * 28)) / 28)
  const jong = code % 28
  const result = [CHOSEONG[cho], JUNGSEONG[jung]]
  if (jong > 0)
    result.push(JONGSEONG[jong])
  return result
}

export interface VisemeFrame {
  blendshapes: ARKitBlendshapes
  durationMs: number
}

export function fillerToVisemeSequence(filler: string, frameMs = 100): VisemeFrame[] {
  const sequence: VisemeFrame[] = []

  for (const char of filler) {
    if (char === ' ') {
      sequence.push({ blendshapes: zeros(), durationMs: Math.floor(frameMs * 0.5) })
      continue
    }

    if (char in VISEME_MAP) {
      const frame = zeros()
      Object.assign(frame, VISEME_MAP[char])
      sequence.push({ blendshapes: frame, durationMs: frameMs })
      continue
    }

    const code = char.charCodeAt(0)
    if (code >= 0xAC00 && code <= 0xD7A3) {
      const phonemes = decomposeHangul(char)
      for (const ph of phonemes) {
        if (ph in VISEME_MAP) {
          const frame = zeros()
          Object.assign(frame, VISEME_MAP[ph])
          const dur = JUNGSEONG.includes(ph) ? frameMs : Math.floor(frameMs * 0.6)
          sequence.push({ blendshapes: frame, durationMs: dur })
        }
      }
    }
    else if (['.', '!', '?', '~'].includes(char)) {
      const frame = zeros()
      Object.assign(frame, VISEME_MAP[char])
      const dur = char === '.' ? Math.floor(frameMs * 0.6) : Math.floor(frameMs * 0.8)
      sequence.push({ blendshapes: frame, durationMs: dur })
    }
  }

  // 마지막에 입 닫기 프레임 추가
  if (sequence.length > 0) {
    const close = zeros()
    close.mouthClose = 0.2
    sequence.push({ blendshapes: close, durationMs: Math.floor(frameMs * 1.5) })
  }

  return sequence
}

// ─────────────────────────────────────────────────
//  눈 깜빡임
// ─────────────────────────────────────────────────

export interface BlinkEvent {
  intervalMs: number
  durationMs: number
  isDouble: boolean
  intensity: number
}

export function generateBlink(emotion: string, energy: number): BlinkEvent {
  let baseInterval = 3500
  if (emotion === 'tired')
    baseInterval = Math.floor(baseInterval * 0.55)
  else if (emotion === 'excited')
    baseInterval = Math.floor(baseInterval * 1.5)
  else if (emotion === 'confused')
    baseInterval = Math.floor(baseInterval * (0.5 + Math.random()))

  const energyFactor = 0.6 + 0.4 * Math.max(0, Math.min(1, energy))
  let interval = Math.floor(baseInterval * energyFactor) + Math.floor(Math.random() * 3001) - 1500
  interval = Math.max(500, interval)

  const durationMs = energy > 0.5 ? 150 : 250
  const isDouble = Math.random() < 0.15
  const intensity = 0.85 + Math.random() * 0.15

  return { intervalMs: interval, durationMs, isDouble, intensity }
}

// ─────────────────────────────────────────────────
//  상태 전환 애니메이션
// ─────────────────────────────────────────────────

export interface TransitionConfig {
  durationMs: number
  easing: 'ease-out' | 'ease-in-out' | 'ease-in'
  overshoot: number
}

const TRANSITION_PRESETS: Record<string, TransitionConfig> = {
  snap: { durationMs: 150, easing: 'ease-out', overshoot: 0.15 },
  smooth: { durationMs: 300, easing: 'ease-in-out', overshoot: 0.0 },
  sluggish: { durationMs: 600, easing: 'ease-in', overshoot: 0.0 },
  jittery: { durationMs: 400, easing: 'ease-out', overshoot: 0.1 },
}

const TRANSITION_RULES: [string, string, string][] = [
  ['happy', 'sassy', 'snap'],
  ['happy', 'confused', 'snap'],
  ['excited', 'sassy', 'snap'],
  ['excited', 'confused', 'snap'],
  ['neutral', 'excited', 'snap'],
  ['*', 'tired', 'sluggish'],
  ['tired', '*', 'sluggish'],
  ['*', 'confused', 'jittery'],
  ['confused', 'confused', 'jittery'],
]

export function pickTransition(prev: string, curr: string, energy: number): TransitionConfig {
  for (const [p, c, name] of TRANSITION_RULES) {
    if ((p === prev || p === '*') && (c === curr || c === '*'))
      return TRANSITION_PRESETS[name]
  }
  if (energy < 0.3)
    return TRANSITION_PRESETS.sluggish
  return TRANSITION_PRESETS.smooth
}

// ─────────────────────────────────────────────────
//  미세 표정 (Micro-Expressions)
// ─────────────────────────────────────────────────

export interface MicroExpression {
  blendshapes: Partial<ARKitBlendshapes>
  durationMs: number
  delayMs: number
}

const MICRO_LIBRARY: Record<string, MicroExpression> = {
  praise_eyewide: {
    blendshapes: { eyeWideLeft: 0.3, eyeWideRight: 0.25, browInnerUp: 0.2 },
    durationMs: 200, delayMs: 0,
  },
  scold_flinch: {
    blendshapes: { eyeSquintLeft: 0.3, eyeSquintRight: 0.3, browDownLeft: 0.2, browDownRight: 0.2 },
    durationMs: 150, delayMs: 0,
  },
  joke_smirk: {
    blendshapes: { mouthSmileLeft: 0.3, mouthDimpleLeft: 0.15, cheekSquintLeft: 0.1 },
    durationMs: 250, delayMs: 0,
  },
  ignore_droop: {
    blendshapes: { browDownLeft: 0.15, browDownRight: 0.15, eyeLookDownLeft: 0.2, eyeLookDownRight: 0.2 },
    durationMs: 300, delayMs: 0,
  },
  dissonance_flash: {
    blendshapes: { eyeWideLeft: 0.2, eyeWideRight: 0.2, browInnerUp: 0.15 },
    durationMs: 180, delayMs: 0,
  },
  surprise_brow: {
    blendshapes: { browInnerUp: 0.4, browOuterUpLeft: 0.3, browOuterUpRight: 0.3, eyeWideLeft: 0.2, eyeWideRight: 0.2 },
    durationMs: 200, delayMs: 0,
  },
}

// ─────────────────────────────────────────────────
//  PATCH-1: Blendshape 레이어 합산 시스템
//
//  Layer 0  base_emotion  additive 1.0  — 감정 확률 블렌딩 결과
//  Layer 1  energy_mod    additive      — 에너지/부조화 보정
//  Layer 2  micro         additive      — 미세 표정 (fade-out 포함)
//  Layer 3  viseme        override      — filler/TTS 입모양 (mouth/jaw/cheek/tongue만)
//  Layer 4  blink         override      — 깜빡임 (eyeBlink* 만)
// ─────────────────────────────────────────────────

export interface BlendLayer {
  id: string
  shapes: Partial<Record<ARKitKey, number>>
  mode: 'additive' | 'override'
  weight: number   // 0–1, override 모드에서 보간용
  active: boolean
}

// viseme 레이어가 override하는 키 집합
const VISEME_OVERRIDE_KEYS = new Set<ARKitKey>([
  'mouthClose', 'mouthFunnel', 'mouthPucker',
  'mouthLeft', 'mouthRight',
  'mouthSmileLeft', 'mouthSmileRight',
  'mouthFrownLeft', 'mouthFrownRight',
  'mouthDimpleLeft', 'mouthDimpleRight',
  'mouthStretchLeft', 'mouthStretchRight',
  'mouthRollLower', 'mouthRollUpper',
  'mouthShrugLower', 'mouthShrugUpper',
  'mouthPressLeft', 'mouthPressRight',
  'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight',
  'jawForward', 'jawLeft', 'jawRight', 'jawOpen',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'tongueOut',
])

// blink 레이어가 override하는 키
const BLINK_OVERRIDE_KEYS = new Set<ARKitKey>(['eyeBlinkLeft', 'eyeBlinkRight'])

// ─────────────────────────────────────────────────
//  PATCH-6: VRM 모델 호환성 분기
//
//  퍼펙트싱크(Perfect Sync) VRM: ARKit 52 커스텀 expression 키를 그대로 사용.
//  일반 VRM: VRM 표준 프리셋 expression (happy/angry/surprised/neutral/relaxed)으로 폴백.
// ─────────────────────────────────────────────────

// 퍼펙트싱크 감지용 키 — 이 중 하나라도 expressionManager에 있으면 퍼펙트싱크 VRM
const DETECT_KEYS: ARKitKey[] = ['eyeBlinkLeft', 'mouthSmileLeft', 'browInnerUp']

// 비퍼펙트싱크 VRM용: EngineEmotion → VRM 표준 프리셋 expression 가중치 매핑
const PRESET_MAP: Record<string, Record<string, number>> = {
  happy: { happy: 0.8 },
  excited: { happy: 1.0 },
  sassy: { angry: 0.5 },
  tired: { neutral: 0.6, relaxed: 0.3 },
  confused: { surprised: 0.4 },
  neutral: { neutral: 0.5 },
}

/**
 * VRM에 퍼펙트싱크 ARKit 블렌드셰이프 키가 등록되어 있는지 감지.
 * DETECT_KEYS 중 하나라도 expressionManager에 등록되어 있으면 퍼펙트싱크로 판단.
 */
export function detectPerfectSync(vrm: VRMCore): boolean {
  const manager = vrm.expressionManager
  if (!manager)
    return false
  return DETECT_KEYS.some(key => manager.getExpression(key) !== null)
}

/**
 * 레이어 배열을 합산하여 최종 52개 blendshape 값을 계산.
 * additive 레이어: 누적 후 최종 단계에서 한 번만 클램핑.
 * override 레이어: 해당 키만 weight 기반 lerp로 대체.
 * 레이어 배열 순서가 우선순위 (뒤로 갈수록 높음).
 */
export function composeLayers(layers: BlendLayer[]): ARKitBlendshapes {
  const result = Object.fromEntries(ARKIT_KEYS.map(k => [k, 0])) as ARKitBlendshapes

  for (const layer of layers) {
    if (!layer.active)
      continue
    for (const [key, value] of Object.entries(layer.shapes) as [ARKitKey, number][]) {
      if (layer.mode === 'additive') {
        // 클램핑은 모든 레이어 처리 후 최종 단계에서 수행
        result[key] += value * layer.weight
      }
      else {
        // override: 해당 키만 lerp로 대체
        result[key] = result[key] * (1 - layer.weight) + value * layer.weight
      }
    }
  }

  // 최종 클램핑
  for (const k of ARKIT_KEYS)
    result[k] = Math.min(1, Math.max(0, result[k]))

  return result
}

// ─────────────────────────────────────────────────
//  ARKit Face Controller 출력 타입
// ─────────────────────────────────────────────────

export interface ARKitFaceOutput {
  blendshapes: ARKitBlendshapes
  transition: TransitionConfig
  microExpressions: MicroExpression[]
  blink: BlinkEvent
  visemeSequence: VisemeFrame[] | null
}

// ─────────────────────────────────────────────────
//  메인 컨트롤러 composable (PATCH-1 레이어 기반)
// ─────────────────────────────────────────────────

export function useARKitFaceController(vrm?: VRMCore) {
  let prevEmotion = 'neutral'
  // PATCH-6: 현재 감정 상태 캐시 (tick()에서 프리셋 폴백 적용 시 사용)
  let currentState = 'neutral'
  // PATCH-6: 퍼펙트싱크 감지 결과 (init(vrm) 호출 시 설정)
  const isPerfectSync = ref(false)

  // ── 레이어 상태 ──
  const layerBase: BlendLayer = { id: 'base_emotion', shapes: {}, mode: 'additive', weight: 1.0, active: true }
  const layerEnergy: BlendLayer = { id: 'energy_mod', shapes: {}, mode: 'additive', weight: 1.0, active: true }
  const layerMicro: BlendLayer = { id: 'micro', shapes: {}, mode: 'additive', weight: 0.0, active: false }
  const layerViseme: BlendLayer = { id: 'viseme', shapes: {}, mode: 'override', weight: 0.0, active: false }
  const layerBlink: BlendLayer = { id: 'blink', shapes: {}, mode: 'override', weight: 0.0, active: false }

  const LAYERS: BlendLayer[] = [layerBase, layerEnergy, layerMicro, layerViseme, layerBlink]

  // micro 레이어 타이머
  let microTimer: ReturnType<typeof setTimeout> | null = null
  let microFadeTimer: ReturnType<typeof setTimeout> | null = null
  const MICRO_FADE_MS = 80

  /**
   * EmotionEngineResult → ARKit 퍼펙트싱크 출력.
   * vrm이 주어지면 expressionManager에 즉시 적용.
   */
  function update(result: ARKitEmotionInput): ARKitFaceOutput {
    const state = result.state
    const probs = result.probabilities
    const { energy, dissonance, filler } = result

    // Layer 0: base_emotion — 감정 확률 블렌딩
    layerBase.shapes = _blendEmotionProbs(probs)
    layerBase.active = true

    // Layer 1: energy_mod — 에너지 보정 delta + 부조화 jitter
    layerEnergy.shapes = _buildEnergyLayer(energy, dissonance)
    layerEnergy.active = true

    // Layer 2: micro — 감정 전환 시 미세 표정 트리거
    _triggerMicros(result)

    // Layer 3: viseme — filler 비지음 (viseme-player.ts가 관리,
    //   여기서는 즉각 적용이 필요할 때 직접 세팅도 지원)
    // Layer 4: blink — generateBlink 결과는 외부에서 rAF 기반으로 관리

    // 레이어 합산
    const composed = composeLayers(LAYERS)

    // 전환 애니메이션
    const transition = pickTransition(prevEmotion, state, energy)

    // 깜빡임 파라미터 (실제 애니는 viseme-player 또는 VRMModel의 blink 루프가 담당)
    const blink = generateBlink(state, energy)

    // 추임새 비지음 시퀀스
    const visemeSequence = filler ? fillerToVisemeSequence(filler) : null

    // VRM에 즉시 적용
    if (vrm?.expressionManager)
      _applyToVRM(vrm, composed, state)

    prevEmotion = state
    currentState = state

    return {
      blendshapes: _roundShapes(composed),
      transition,
      microExpressions: layerMicro.active ? [{ blendshapes: layerMicro.shapes as Partial<ARKitBlendshapes>, durationMs: 200, delayMs: 0 }] : [],
      blink,
      visemeSequence,
    }
  }

  // ── 확률 블렌딩 (Layer 0) ──
  function _blendEmotionProbs(probs: Record<string, number>): Partial<ARKitBlendshapes> {
    const result = zeros()
    for (const [emotionKey, weight] of Object.entries(probs)) {
      const base = BASE_EXPRESSIONS[emotionKey] ?? BASE_EXPRESSIONS.neutral
      for (const key of ARKIT_KEYS)
        result[key] += (base[key] ?? 0.0) * weight
    }
    return result
  }

  // ── 에너지/부조화 보정 delta (Layer 1) ──
  // additive이므로 "base에서의 변화량"만 담는다.
  function _buildEnergyLayer(energy: number, dissonance: number): Partial<ARKitBlendshapes> {
    const e = Math.max(0, Math.min(1, energy))
    const delta: Partial<ARKitBlendshapes> = {}

    // 피로 눈 감김 증가
    const fatigueBlink = (1 - e) * 0.35
    delta.eyeBlinkLeft = fatigueBlink
    delta.eyeBlinkRight = fatigueBlink + 0.03

    // 눈썹 내림 증가
    delta.browDownLeft = (1 - e) * 0.1
    delta.browDownRight = (1 - e) * 0.1

    // 입꼬리 내림
    delta.mouthFrownLeft = (1 - e) * 0.1
    delta.mouthFrownRight = (1 - e) * 0.1

    // 시선 아래
    delta.eyeLookDownLeft = (1 - e) * 0.2
    delta.eyeLookDownRight = (1 - e) * 0.2

    // 부조화 jitter (dissonance > 0.4 시)
    if (dissonance > 0.4) {
      const jitterKeys: ARKitKey[] = [
        'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
        'eyeSquintLeft', 'eyeSquintRight',
        'mouthSmileLeft', 'mouthSmileRight',
        'jawLeft', 'jawRight',
      ]
      const scale = dissonance * 0.06
      for (const key of jitterKeys)
        delta[key] = (delta[key] ?? 0) + (Math.random() - 0.5) * scale
    }

    return delta
  }

  // ── 미세 표정 트리거 (Layer 2) ──
  function _triggerMicros(result: ARKitEmotionInput): void {
    const state = result.state
    const prev = result.prevState

    let micro: Partial<ARKitBlendshapes> | null = null
    let durationMs = 200

    if (prev !== state) {
      if (state === 'happy' || state === 'excited') {
        micro = MICRO_LIBRARY.praise_eyewide.blendshapes as Partial<ARKitBlendshapes>
        durationMs = MICRO_LIBRARY.praise_eyewide.durationMs
      }
      else if (state === 'sassy' && (prev === 'happy' || prev === 'excited')) {
        micro = MICRO_LIBRARY.scold_flinch.blendshapes as Partial<ARKitBlendshapes>
        durationMs = MICRO_LIBRARY.scold_flinch.durationMs
      }
      else if (state === 'confused') {
        micro = MICRO_LIBRARY.surprise_brow.blendshapes as Partial<ARKitBlendshapes>
        durationMs = MICRO_LIBRARY.surprise_brow.durationMs
      }
    }
    if (result.dissonance > 0.6) {
      micro = MICRO_LIBRARY.dissonance_flash.blendshapes as Partial<ARKitBlendshapes>
      durationMs = MICRO_LIBRARY.dissonance_flash.durationMs
    }

    if (!micro)
      return

    // 기존 micro 타이머 정리
    if (microTimer !== null)
      clearTimeout(microTimer)
    if (microFadeTimer !== null)
      clearTimeout(microFadeTimer)

    layerMicro.shapes = micro
    layerMicro.weight = 1.0
    layerMicro.active = true

    // duration 후 fade-out 시작
    microTimer = setTimeout(() => {
      // PATCH-1: micro-expression fade-out (~80ms linear)
      // weight를 0으로 lerp 후 비활성화 (하드컷 방지)
      const steps = 4
      let step = 0
      const fadeInterval = MICRO_FADE_MS / steps
      const fadeStep = () => {
        step++
        layerMicro.weight = Math.max(0, 1 - step / steps)
        if (step >= steps) {
          layerMicro.active = false
          layerMicro.weight = 0
        }
        else {
          microFadeTimer = setTimeout(fadeStep, fadeInterval)
        }
      }
      microFadeTimer = setTimeout(fadeStep, fadeInterval)
    }, durationMs)
  }

  // ── viseme 레이어 직접 세팅 (viseme-player와 연동) ──
  function setVisemeLayer(shapes: Partial<ARKitBlendshapes> | null, weight = 1.0): void {
    if (!shapes) {
      layerViseme.active = false
      layerViseme.weight = 0
      return
    }
    // viseme 레이어는 mouth/jaw/cheek/tongue 키만
    const filtered: Partial<ARKitBlendshapes> = {}
    for (const [k, v] of Object.entries(shapes) as [ARKitKey, number][]) {
      if (VISEME_OVERRIDE_KEYS.has(k))
        filtered[k] = v
    }
    layerViseme.shapes = filtered
    layerViseme.weight = weight
    layerViseme.active = true
  }

  // ── blink 레이어 직접 세팅 (blink 루프와 연동) ──
  function setBlinkLayer(intensity: number): void {
    layerBlink.shapes = {
      eyeBlinkLeft: intensity,
      eyeBlinkRight: intensity,
    }
    // blink 레이어는 eyeBlink* 키만 override
    for (const key of Object.keys(layerBlink.shapes) as ARKitKey[]) {
      if (!BLINK_OVERRIDE_KEYS.has(key))
        delete layerBlink.shapes[key]
    }
    layerBlink.weight = 1.0
    layerBlink.active = true
  }

  function clearBlinkLayer(): void {
    layerBlink.active = false
    layerBlink.weight = 0
  }

  // ── VRM expressionManager 적용 ──
  // NOTICE: VRM 1.0 표준 expression은 ARKit 키와 다름.
  //         퍼펙트싱크 지원 VRM은 커스텀 expression으로 ARKit 52 키를 그대로 등록해야 함.
  //         지원 안 되는 키는 무시됨 (getExpression이 null이면 setValue 건너뜀).
  // PATCH-6: isPerfectSync에 따라 ARKit 직접 적용 또는 프리셋 폴백 분기.
  function _applyToVRM(vrmInstance: VRMCore, shapes: ARKitBlendshapes, state = currentState): void {
    const manager = vrmInstance.expressionManager
    if (!manager)
      return

    if (isPerfectSync.value) {
      for (const key of ARKIT_KEYS) {
        const val = shapes[key]
        if (val > 0.001 && manager.getExpression(key) !== null)
          manager.setValue(key, val)
      }
    }
    else {
      _applyPresetFallback(manager, state)
    }
  }

  /**
   * PATCH-6: 비퍼펙트싱크 VRM용 프리셋 폴백.
   * PRESET_MAP에서 현재 감정에 해당하는 VRM 표준 expression을 적용.
   */
  function _applyPresetFallback(
    manager: NonNullable<VRMCore['expressionManager']>,
    state: string,
  ): void {
    const presets = PRESET_MAP[state] ?? PRESET_MAP.neutral
    for (const [name, weight] of Object.entries(presets)) {
      if (manager.getExpression(name) !== null)
        manager.setValue(name, weight)
    }
  }

  function _roundShapes(shapes: ARKitBlendshapes): ARKitBlendshapes {
    return Object.fromEntries(
      ARKIT_KEYS.map(k => [k, Math.round(shapes[k] * 10000) / 10000]),
    ) as ARKitBlendshapes
  }

  /** rAF 루프에서 호출 — 레이어 합산 결과를 VRM에 즉시 적용 */
  function tick(): ARKitBlendshapes {
    const composed = composeLayers(LAYERS)
    if (vrm?.expressionManager)
      _applyToVRM(vrm, composed, currentState)
    return composed
  }

  function dispose(): void {
    if (microTimer !== null)
      clearTimeout(microTimer)
    if (microFadeTimer !== null)
      clearTimeout(microFadeTimer)
  }

  /**
   * PATCH-6: VRM 로드 후 호출하여 퍼펙트싱크 여부를 감지하고 캐싱.
   * VRMModel.vue의 onVRMLoaded 콜백에서 한 번 호출하면 됨.
   */
  function init(vrmInstance: VRMCore): void {
    isPerfectSync.value = detectPerfectSync(vrmInstance)
  }

  return {
    update,
    tick,
    init,
    setVisemeLayer,
    setBlinkLayer,
    clearBlinkLayer,
    composeLayers: () => composeLayers(LAYERS),
    dispose,
    isPerfectSync,
    // 레이어 직접 접근 (디버그용)
    layers: LAYERS,
  }
}
