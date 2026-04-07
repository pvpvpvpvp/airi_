/**
 * VRM Viseme Sequence Player
 * ===========================
 * ARKit 52 blendshape 기반 viseme 시퀀스를 VRM expressionManager에 적용.
 *
 * 퍼펙트싱크 VRM이 아닌 표준 VRM에서는 ARKit 입 관련 키를
 * aa / ee / ih / oh / ou 5개 VRM 표준 expression으로 근사 매핑.
 *
 * 사용법:
 *   const { playVisemeSequence, stopVisemeSequence } = useVRMVisemePlayer(vrmRef)
 *   // 추임새 표시 전:
 *   playVisemeSequence(fillerToVisemeSequence(result.filler))
 *   // TTS 시작 시 중단 (wLipSync가 대신 제어):
 *   stopVisemeSequence()
 */

import type { VRMCore } from '@pixiv/three-vrm-core'
import type { Ref } from 'vue'

import type { ARKitBlendshapes, VisemeFrame } from './arkit-face'

// ─────────────────────────────────────────────────
//  ARKit 입 키 → VRM 표준 expression 근사 매핑
//
//  NOTICE: VRM 1.0 표준 expression은 aa/ee/ih/oh/ou 5개.
//  퍼펙트싱크 VRM은 ARKit 52 키를 그대로 쓰므로 그쪽은 arkit-face.ts의
//  applyToVRM()이 직접 처리함. 여기서는 표준 VRM용 근사값을 계산.
// ─────────────────────────────────────────────────

const VRM_MOUTH_KEYS = ['aa', 'ee', 'ih', 'oh', 'ou'] as const
type VRMMouthKey = typeof VRM_MOUTH_KEYS[number]

/**
 * ARKit blendshapes → VRM 표준 mouth expression 근사 변환.
 * 입 벌림/모양 키들을 가중합해서 aa/ee/ih/oh/ou로 매핑.
 */
function arkitToVRMMouth(shapes: ARKitBlendshapes): Record<VRMMouthKey, number> {
  const jaw = shapes.jawOpen ?? 0
  const funnel = shapes.mouthFunnel ?? 0
  const pucker = shapes.mouthPucker ?? 0
  const smileL = shapes.mouthSmileLeft ?? 0
  const smileR = shapes.mouthSmileRight ?? 0
  const stretchL = shapes.mouthStretchLeft ?? 0
  const stretchR = shapes.mouthStretchRight ?? 0
  const lowerDownL = shapes.mouthLowerDownLeft ?? 0
  const lowerDownR = shapes.mouthLowerDownRight ?? 0

  const smile = (smileL + smileR) / 2
  const stretch = (stretchL + stretchR) / 2
  const lowerDown = (lowerDownL + lowerDownR) / 2

  return {
    // aa: 입을 크게 벌림 — jawOpen + mouthLowerDown 위주
    aa: Math.min(1, jaw * 0.8 + lowerDown * 0.4),
    // ee: 옆으로 당긴 입 — smile + stretch
    ee: Math.min(1, smile * 0.7 + stretch * 0.5),
    // ih: 약간 벌린 중립 — jawOpen 약하게 + stretch
    ih: Math.min(1, jaw * 0.4 + stretch * 0.3),
    // oh: 둥근 입 — funnel + pucker + jawOpen 약하게
    oh: Math.min(1, funnel * 0.6 + pucker * 0.5 + jaw * 0.3),
    // ou: 오므린 입 — pucker 위주
    ou: Math.min(1, pucker * 0.8 + funnel * 0.3),
  }
}

// ─────────────────────────────────────────────────
//  Viseme Player composable
// ─────────────────────────────────────────────────

export function useVRMVisemePlayer(vrmRef: Ref<VRMCore | undefined>, isPerfectSync?: Ref<boolean>) {
  let playbackTimer: ReturnType<typeof setTimeout> | null = null
  let currentIndex = 0
  let activeSequence: VisemeFrame[] = []
  let isPlaying = false

  function stopVisemeSequence(): void {
    if (playbackTimer !== null) {
      clearTimeout(playbackTimer)
      playbackTimer = null
    }
    isPlaying = false
    activeSequence = []
    currentIndex = 0
    _clearMouth()
  }

  function _clearMouth(): void {
    const vrm = vrmRef.value
    if (!vrm?.expressionManager)
      return
    for (const key of VRM_MOUTH_KEYS)
      vrm.expressionManager.setValue(key, 0)
  }

  function _applyFrame(frame: VisemeFrame): void {
    const vrm = vrmRef.value
    if (!vrm?.expressionManager)
      return

    // PATCH-6: isPerfectSync ref 주입 시 사용, 없으면 jawOpen 존재 여부로 폴백
    const perfectSync = isPerfectSync?.value ?? (vrm.expressionManager.getValue('jawOpen') !== null)
    if (perfectSync) {
      for (const [key, val] of Object.entries(frame.blendshapes)) {
        if (val > 0.001)
          vrm.expressionManager.setValue(key, val)
      }
      return
    }

    // 표준 VRM: ARKit → aa/ee/ih/oh/ou 근사 변환
    const vrmMouth = arkitToVRMMouth(frame.blendshapes as ARKitBlendshapes)
    for (const key of VRM_MOUTH_KEYS)
      vrm.expressionManager.setValue(key, vrmMouth[key])
  }

  function _scheduleNext(): void {
    if (!isPlaying || currentIndex >= activeSequence.length) {
      _clearMouth()
      isPlaying = false
      return
    }

    const frame = activeSequence[currentIndex]
    _applyFrame(frame)
    currentIndex++

    playbackTimer = setTimeout(_scheduleNext, frame.durationMs)
  }

  /**
   * viseme 시퀀스 재생 시작.
   * 이미 재생 중이면 중단 후 새 시퀀스로 교체.
   */
  function playVisemeSequence(sequence: VisemeFrame[]): void {
    stopVisemeSequence()
    if (sequence.length === 0)
      return

    activeSequence = sequence
    currentIndex = 0
    isPlaying = true
    _scheduleNext()
  }

  return {
    playVisemeSequence,
    stopVisemeSequence,
    isPlaying: () => isPlaying,
  }
}
