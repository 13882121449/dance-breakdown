/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端 API 基地址，默认 http://localhost:8000 */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
