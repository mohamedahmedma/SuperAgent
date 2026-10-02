/// <reference types="vite/client" />

/**
 * The settings this app reads, declared in one place.
 *
 * All three live in the repo-root `.env` (see `vite.config.ts`'s `envDir`) and are baked
 * into the bundle at build time — changing one means rebuilding, not restarting.
 *
 * `vite/client` already declares an index signature, so this does not make a typo fail to
 * compile. It is here to be the list: the answer to "what can the frontend be configured
 * with" should not be a grep for `import.meta.env`.
 */
interface ImportMetaEnv {
  /** The chat backend's origin. Empty = same origin as the app (the default). */
  readonly VITE_API_BASE_URL?: string;
  /** The identity service's origin. Empty = same origin as the app (the default). */
  readonly VITE_IDENTITY_BASE_URL?: string;
  /** Which school this login page belongs to. Empty for a single-school deployment. */
  readonly VITE_SCHOOL_CODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}
