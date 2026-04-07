import type { SystemMessage } from '@xsai/shared-chat'

import { EMOTION_EmotionMotionName_value, EMOTION_VALUES } from '../emotions'

/**
 * 기본 시스템 프롬프트 생성.
 * @param prefix - 캐릭터 설정 앞부분
 * @param suffix - 캐릭터 설정 뒷부분
 * @param emotionEnginePrompt - useEmotionEngineStore().buildEmotionPrompt() 출력 (선택)
 *   있으면 감정 상태 블록이 suffix 뒤에 추가됨.
 */
function message(prefix: string, suffix: string, emotionEnginePrompt?: string) {
  const parts = [
    prefix,
    EMOTION_VALUES
      .map(emotion => `- ${emotion} (Emotion for feeling ${EMOTION_EmotionMotionName_value[emotion]})`)
      .join('\n'),
    suffix,
  ]

  if (emotionEnginePrompt)
    parts.push(emotionEnginePrompt)

  return {
    role: 'system',
    content: parts.join('\n\n'),
  } satisfies SystemMessage
}

export default message
