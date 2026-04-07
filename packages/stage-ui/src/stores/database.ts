/**
 * DuckDB 싱글턴 스토어
 * ================================================================
 * @proj-airi/drizzle-duckdb-wasm 인스턴스를 앱 전역에서 공유한다.
 * Stage.vue의 onMounted에서 setDb()로 등록하고,
 * memory-engine.ts 같은 Pinia store에서 useDatabaseStore()로 접근한다.
 */

import type { DuckDBWasmDrizzleDatabase } from '@proj-airi/drizzle-duckdb-wasm'

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

export const useDatabaseStore = defineStore('database', () => {
  const db = ref<DuckDBWasmDrizzleDatabase | null>(null)

  const isReady = computed(() => db.value !== null)

  function setDb(instance: DuckDBWasmDrizzleDatabase): void {
    db.value = instance
  }

  /**
   * SQL 실행 헬퍼 — DDL/DML/SELECT 공통.
   * DB가 아직 초기화되지 않은 경우 빈 배열을 반환한다.
   */
  async function execute(sql: string): Promise<Record<string, unknown>[]> {
    if (!db.value)
      return []
    return (await db.value.execute(sql)) as Record<string, unknown>[]
  }

  return { db, isReady, setDb, execute }
})
