/**
 * xoshiro128+ — 시드 기반 PRNG.
 * splitmix32로 시드를 4-state로 확장한 뒤 xoshiro128+ 알고리즘으로 난수 생성.
 * 주기: 2^128 - 1. 게임/시뮬레이션 용도에 충분.
 * PATCH-3 직렬화를 위한 getState() / setState() 포함.
 */
export class SeededRandom {
  private s: [number, number, number, number]

  constructor(seed: number = Date.now()) {
    this.s = [0, 0, 0, 0]
    this.seed(seed)
  }

  /** 시드 재설정 */
  seed(seed: number): void {
    let z = seed | 0
    const splitmix = (): number => {
      z = (z + 0x9E3779B9) | 0
      let t = z ^ (z >>> 16)
      t = Math.imul(t, 0x21F0AAAD)
      t = t ^ (t >>> 15)
      t = Math.imul(t, 0x735A2D97)
      return (t ^ (t >>> 15)) >>> 0
    }
    this.s[0] = splitmix()
    this.s[1] = splitmix()
    this.s[2] = splitmix()
    this.s[3] = splitmix()
  }

  /** 0 이상 1 미만 균등분포 */
  random(): number {
    const s = this.s
    const result = (s[0] + s[3]) >>> 0
    const t = s[1] << 9

    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    // rotl(s[3], 11)
    s[3] = (s[3] << 11) | (s[3] >>> 21)

    return result / 4294967296
  }

  /** 내부 상태 스냅샷 (직렬화용) */
  getState(): [number, number, number, number] {
    return [...this.s] as [number, number, number, number]
  }

  /** 내부 상태 복원 (역직렬화용) */
  setState(state: [number, number, number, number]): void {
    this.s = [...state] as [number, number, number, number]
  }
}
