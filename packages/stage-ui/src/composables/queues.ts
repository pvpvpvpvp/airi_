import type { UseQueueReturn } from '@proj-airi/stream-kit'

import type { Emotion, EmotionPayload } from '../constants/emotions'
import type { EngineEmotion, EngineEvent } from '../stores/modules/emotion-engine'

import { sleep } from '@moeru/std'
import { createQueue } from '@proj-airi/stream-kit'

import { EMOTION_VALUES } from '../constants/emotions'
import { ENGINE_EMOTION_VALUES, ENGINE_EVENT_VALUES } from '../stores/modules/emotion-engine'

export function useEmotionsMessageQueue(
  emotionsQueue: UseQueueReturn<EmotionPayload>,
  onEngineEvent?: (event: EngineEvent) => void,
  // PATCH-5: 기존 emotion ACT를 받아 setEmotionDirect()로 연결
  onEmotionDirect?: (emotion: EngineEmotion) => void,
) {
  const normalizeEmotionName = (value: string): Emotion | null => {
    const normalized = value.trim().toLowerCase()
    if (EMOTION_VALUES.includes(normalized as Emotion))
      return normalized as Emotion
    return null
  }

  const normalizeIntensity = (value: unknown): number => {
    if (typeof value !== 'number' || Number.isNaN(value))
      return 1
    return Math.min(1, Math.max(0, value))
  }

  const normalizeEngineEvent = (value: string): EngineEvent | null => {
    const normalized = value.trim().toLowerCase()
    if (ENGINE_EVENT_VALUES.includes(normalized as EngineEvent))
      return normalized as EngineEvent
    return null
  }

  const normalizeEngineEmotion = (value: string): EngineEmotion | null => {
    const normalized = value.trim().toLowerCase()
    if (ENGINE_EMOTION_VALUES.includes(normalized as EngineEmotion))
      return normalized as EngineEmotion
    return null
  }

  function parseActPayload(content: string) {
    const match = /<\|ACT\s*(?::\s*)?(\{[\s\S]*\})\|>/i.exec(content)
    if (!match)
      return { ok: false, emotion: null as EmotionPayload | null, event: null as EngineEvent | null }

    const payloadText = match[1]
    try {
      const payload = JSON.parse(payloadText) as { emotion?: unknown, event?: unknown }

      // event 필드 처리 — 감정 엔진 이벤트 (praise/scold/joke 등)
      if (typeof payload.event === 'string') {
        const evt = normalizeEngineEvent(payload.event)
        if (evt)
          return { ok: true, emotion: null, event: evt }
      }

      // emotion 필드 처리 — 기존 방식 유지 (하위 호환)
      const emotion = payload?.emotion
      if (typeof emotion === 'string') {
        const normalized = normalizeEmotionName(emotion)
        if (normalized)
          return { ok: true, emotion: { name: normalized, intensity: 1 }, event: null }
      }
      else if (emotion && typeof emotion === 'object' && !Array.isArray(emotion)) {
        if ('name' in emotion && typeof (emotion as { name?: unknown }).name === 'string') {
          const normalized = normalizeEmotionName((emotion as { name: string }).name)
          if (normalized) {
            const intensity = normalizeIntensity((emotion as { intensity?: unknown }).intensity)
            return { ok: true, emotion: { name: normalized, intensity }, event: null }
          }
        }
      }
    }
    catch (e) {
      console.warn(`[parseActPayload] Failed to parse ACT payload JSON: "${payloadText}"`, e)
    }

    return { ok: false, emotion: null as EmotionPayload | null, event: null as EngineEvent | null }
  }

  return createQueue<string>({
    handlers: [
      async (ctx) => {
        const actParsed = parseActPayload(ctx.data)
        if (!actParsed.ok)
          return

        if (actParsed.event) {
          // 감정 엔진 이벤트 경로
          ctx.emit('engine-event', actParsed.event)
          onEngineEvent?.(actParsed.event)
        }
        else if (actParsed.emotion) {
          // PATCH-5: 기존 emotion ACT — 디스플레이 큐에 enqueue하면서
          // 동시에 엔진에도 setEmotionDirect로 알림 (상태 동기화)
          ctx.emit('emotion', actParsed.emotion)
          emotionsQueue.enqueue(actParsed.emotion)
          const engineEmo = normalizeEngineEmotion(actParsed.emotion.name)
          if (engineEmo)
            onEmotionDirect?.(engineEmo)
        }
      },
    ],
  })
}

export function useDelayMessageQueue() {
  function splitDelays(content: string) {
    if (!(/<\|DELAY:\d+\|>/i.test(content))) {
      return {
        ok: false,
        delay: 0,
      }
    }

    const delayExecArray = /<\|DELAY:(\d+)\|>/i.exec(content)

    const delay = delayExecArray?.[1]
    if (!delay) {
      return {
        ok: false,
        delay: 0,
      }
    }

    const delaySeconds = Number.parseFloat(delay)

    if (delaySeconds <= 0 || Number.isNaN(delaySeconds)) {
      return {
        ok: true,
        delay: 0,
      }
    }

    return {
      ok: true,
      delay: delaySeconds,
    }
  }

  return createQueue<string>({
    handlers: [
      async (ctx) => {
        const { ok, delay } = splitDelays(ctx.data)
        if (ok) {
          ctx.emit('delay', delay)
          await sleep(delay * 1000)
        }
      },
    ],
  })
}
