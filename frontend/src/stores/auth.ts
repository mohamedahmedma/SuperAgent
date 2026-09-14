import { defineStore } from 'pinia';
import identityApi, {
  ACCESS_TOKEN_KEY,
  REFRESH_TOKEN_KEY,
} from '@/utils/identityApi';
import {
  clearStoredTokens,
  exchangeRefreshToken,
  isStale,
  readStoredTokens,
  withRefreshLock,
  writeStoredTokens,
} from '@/utils/session';
import type { SessionTokens } from '@/utils/session';
import type { CurrentUser, UserRole } from '@/types/user';

/**
 * Where a WhatsApp verification has got to.
 *
 *   idle       nothing started
 *   waiting    the parent has been given a link and has not sent it yet
 *   code_sent  the school replied with a code; the parent is typing it
 *   failed     the challenge is dead — expired, refused, or the number is unknown
 *
 * Mirrors identity's own vocabulary except for `idle`, which only exists in the UI, and
 * `failed`, which flattens identity's `rejected` together with the local timeout. A
 * parent cannot act differently on those two, and pretending otherwise would put a
 * distinction on screen that means nothing to them.
 */
export type WhatsAppStatus = 'idle' | 'waiting' | 'code_sent' | 'failed';

/**
 * The poll timer, deliberately outside the store.
 *
 * Pinia state is reactive, and a timer id is a number nothing should ever re-render on.
 * Keeping it module-scoped also means `stopPolling` works even if the component that
 * started it has already been torn down — which is exactly when it matters, because a
 * poller that outlives its panel keeps hitting identity forever.
 */
let pollTimer: ReturnType<typeof setInterval> | null = null;

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

/** How often to ask. The parent is switching apps, so this is not a race. */
const POLL_INTERVAL_MS = 2000;

/**
 * The one refresh in flight, if any. Module-scoped like the poll timer, and for the same
 * reason: a promise is not something to re-render on. Every caller that finds the token
 * stale at the same moment — the stream, an image, a sessions fetch — waits on this one
 * exchange instead of spending the refresh token three times.
 */
let refreshInFlight: Promise<string> | null = null;

/**
 * Pull the machine-readable code out of an identity refusal.
 *
 * The existing handler keeps only `message`, which is right for a password login where
 * every failure reads the same. This flow has several — a wrong code, an expired
 * challenge, a number the school does not hold — and the page shows a different thing for
 * each, so the code has to survive.
 */
function refusal(error: any): { code: string; message: string } {
  const detail = error?.response?.data?.detail;
  if (detail && typeof detail === 'object') {
    return {
      code: String(detail.code || ''),
      message: String(detail.message || error?.message || ''),
    };
  }
  return {
    code: '',
    message: (typeof detail === 'string' ? detail : error?.message) || 'Something went wrong',
  };
}

/**
 * Authentication now talks to the identity service, not the chat backend.
 *
 * The backend no longer has `/auth/*` routes — it only verifies the tokens it is
 * handed. Login, registration and refresh all go to a separate origin, which is why
 * every call here uses `identityApi` rather than the shared `api` instance.
 *
 * ## The session stays open until the parent signs out
 *
 * This store is the authority on the live session. Every caller that needs a token asks
 * it — `ensureFreshToken()` for axios, `authorizedFetch()` for the two raw `fetch` calls —
 * and gets one that will still be valid when the request lands, renewed first if not.
 * Identity rotates the refresh token on every renewal and keeps the session alive for a
 * full inactivity window from each, so a parent who keeps using the app is never asked to
 * sign in again. The session ends in exactly two ways: the parent signs out
 * (`handleLogout`), or identity refuses a renewal (`endSession`).
 */
export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: readStoredTokens().accessToken,
    refreshToken: readStoredTokens().refreshToken,
    currentUser: null as CurrentUser | null,
    authForm: {
      username: '',
      password: '',
    },
    authLoading: false,

    /**
     * A WhatsApp verification in progress.
     *
     * `pollSecret` is this browser's half of the proof and is deliberately NOT persisted
     * to localStorage: it is a short-lived bearer credential, the challenge dies in ten
     * minutes anyway, and a parent who reloads the page should start again rather than
     * resume something they can no longer see the WhatsApp thread for.
     */
    whatsapp: {
      status: 'idle' as WhatsAppStatus,
      pollSecret: '',
      link: '',
      message: '',
      businessNumber: '',
      expiresAt: '',
      /** Her name, once the school's records have recognised the number. */
      displayName: '',
      code: '',
      error: '',
      /** identity's own refusal code, so the page can say something specific. */
      errorCode: '',
      busy: false,
    },
  }),

  getters: {
    isAuthenticated(): boolean {
      return !!this.token && !!this.currentUser;
    },
    isAdmin(): boolean {
      return this.currentUser?.role === 'admin';
    },
    /**
     * Whether to offer the student-records features at all. Not an authorisation
     * check — the records facade makes that decision from the signed token — just a
     * hint about which UI is worth showing.
     */
    isParent(): boolean {
      return !!this.currentUser?.guardianId;
    },
  },

  actions: {
    /**
     * A token good for the request about to be made, renewed first if the one in hand is
     * about to expire. Empty when there is no session — nothing stored, or a renewal
     * identity refused.
     */
    async ensureFreshToken(): Promise<string> {
      if (!isStale(this.token)) return this.token;
      if (!this.refreshToken) return this.token;
      return this.refreshSession();
    },

    /**
     * Exchange the refresh token for new tokens — once, however many callers ask at the
     * same moment, and once across tabs where the browser can arrange that.
     *
     * Resolves to the new access token, or to the current one when identity could not be
     * reached (it may still work, and the next call tries again), or to '' when identity
     * refused — in which case the session has been ended and the app told.
     */
    refreshSession(): Promise<string> {
      if (!refreshInFlight) {
        refreshInFlight = withRefreshLock(() => this.renewTokens()).finally(() => {
          refreshInFlight = null;
        });
      }
      return refreshInFlight;
    },

    async renewTokens(): Promise<string> {
      // Another tab may have renewed while this one waited for the lock, or a moment
      // before it: its tokens are in storage, and adopting them costs nothing.
      const stored = readStoredTokens();
      if (stored.accessToken && stored.accessToken !== this.token && !isStale(stored.accessToken)) {
        this.token = stored.accessToken;
        this.refreshToken = stored.refreshToken || this.refreshToken;
        return this.token;
      }

      const outcome = await exchangeRefreshToken(this.refreshToken);
      if (outcome.status === 'refreshed') {
        this.applyTokens(outcome.tokens);
        return this.token;
      }
      if (outcome.status === 'rejected') {
        this.endSession();
        return '';
      }
      // Identity is unreachable. The token in hand may not have expired yet, and a request
      // made with it may well succeed; a network blip must not sign a parent out.
      return this.token;
    },

    /**
     * `fetch` with a bearer token that is fresh, and one renewal-and-retry should the
     * server refuse it anyway. For the two calls axios cannot make: the chat stream, which
     * reads a `ReadableStream`, and an image, which is fetched as a blob.
     */
    async authorizedFetch(input: string, init: RequestInit = {}): Promise<Response> {
      const send = (token: string) =>
        fetch(input, {
          ...init,
          headers: {
            ...((init.headers as Record<string, string> | undefined) || {}),
            Authorization: `Bearer ${token}`,
          },
        });

      let response = await send(await this.ensureFreshToken());
      if (response.status === 401 && this.refreshToken) {
        const token = await this.refreshSession();
        if (token) {
          response = await send(token);
        }
      }
      return response;
    },

    /**
     * Pick the session up where the browser left it: on page load, with whatever storage
     * holds. A stale access token is renewed before anything is asked of it, so a parent
     * who reopens the app a week later is signed in, not signed out.
     */
    async restoreSession(): Promise<void> {
      if (!this.token && !this.refreshToken) return;
      const token = await this.ensureFreshToken();
      if (!token) return;
      try {
        await this.fetchMe();
      } catch {
        // `fetchMe` has already ended the session; the login panel is showing.
      }
    },

    /**
     * Adopt tokens another tab writes, and sign out when another tab signs out. Returns
     * the function that stops watching.
     */
    watchOtherTabs(): () => void {
      const onStorage = (event: StorageEvent) => {
        if (event.key !== null && event.key !== ACCESS_TOKEN_KEY && event.key !== REFRESH_TOKEN_KEY) {
          return;
        }
        const stored = readStoredTokens();
        if (!stored.accessToken && !stored.refreshToken) {
          if (this.token || this.refreshToken) {
            this.token = '';
            this.refreshToken = '';
            this.currentUser = null;
          }
          return;
        }
        if (stored.accessToken !== this.token || stored.refreshToken !== this.refreshToken) {
          this.token = stored.accessToken;
          this.refreshToken = stored.refreshToken;
        }
      };
      window.addEventListener('storage', onStorage);
      return () => window.removeEventListener('storage', onStorage);
    },

    async fetchMe(retried = false): Promise<void> {
      const token = await this.ensureFreshToken();
      if (!token) return;
      try {
        const response = await identityApi.get('/v1/auth/me', {
          headers: { Authorization: `Bearer ${token}` },
        });
        this.currentUser = {
          username: response.data.username,
          role: response.data.role as UserRole,
          guardianId: response.data.guardian_id ?? null,
          displayName: response.data.display_name || '',
        };
      } catch (error: any) {
        // A token the clock called fresh was refused — the signing key rotated, or the
        // session was revoked. One renewal and one retry; refused again, it is over.
        if (error?.response?.status === 401 && !retried && this.refreshToken) {
          const renewed = await this.refreshSession();
          if (renewed && renewed !== token) {
            return this.fetchMe(true);
          }
        }
        this.handleLogout();
        throw error;
      }
    },

    /** Sign in with a username and password. Staff and administrators only. */
    async handleAuthSubmit() {
      if (this.authLoading) return;
      const username = this.authForm.username.trim();
      const password = this.authForm.password.trim();
      if (!username || !password) {
        throw new Error('Username and password cannot be empty');
      }

      this.authLoading = true;
      try {
        /* Login only. There is no registration endpoint any more: `/v1/auth/register` was
           removed from identity when the shared admin key was, because an administrator is
           now exactly an account with role=admin — so a public route that could mint one
           would be a public route into every family's records.

           Accounts are created by an administrator through POST /v1/admin/accounts, and
           the first administrator is seeded from configuration on every startup. Parents
           never come through here at all; they sign in over WhatsApp. */
        const { data } = await identityApi.post('/v1/auth/login', { username, password });
        this.applySession(data);

        this.authForm.password = '';
      } catch (error: any) {
        // The identity service returns `detail` as an object with a machine-readable
        // `code`; the old backend returned a bare string. Both are handled so a
        // mid-migration deployment cannot render "[object Object]" at a parent.
        const detail = error.response?.data?.detail;
        const message =
          (typeof detail === 'string' ? detail : detail?.message) ||
          error.message ||
          'Authentication failed';
        throw new Error(message);
      } finally {
        this.authLoading = false;
      }
    },

    /**
     * Ask identity for a challenge and start watching for the parent's message.
     *
     * Sends no phone number, because there is none to send: the parent proves which
     * number they hold by messaging from it. That is also why this cannot be used to ask
     * whether a given number belongs to a parent.
     */
    async startWhatsAppLogin(school?: string) {
      if (this.whatsapp.busy) return;
      stopPolling();
      this.whatsapp.busy = true;
      this.whatsapp.error = '';
      this.whatsapp.errorCode = '';
      this.whatsapp.code = '';
      this.whatsapp.displayName = '';

      try {
        /* The school this login page belongs to, where the estate has several. It picks
           the WhatsApp number the link points at, and identity checks it again against the
           number the parent's message actually arrives on — so a link steered at the wrong
           school is refused rather than resolving the parent against a database their
           children are not in. Omitted where the estate holds one school, which is the
           default and needs no configuration. */
        const configured = (import.meta.env.VITE_SCHOOL_CODE as string | undefined) || '';
        const code = (school || configured).trim();
        /* Called with exactly one argument when no school is configured, so a
           single-school estate makes the identical request it always made. */
        const { data } = code
          ? await identityApi.post('/v1/auth/whatsapp/start', undefined, {
              params: { school: code },
            })
          : await identityApi.post('/v1/auth/whatsapp/start');
        this.whatsapp.pollSecret = data.poll_secret;
        this.whatsapp.link = data.link;
        this.whatsapp.message = data.message;
        this.whatsapp.businessNumber = data.business_number;
        this.whatsapp.expiresAt = data.expires_at;
        this.whatsapp.status = 'waiting';
        this.beginWhatsAppPolling();
      } catch (error: any) {
        const { message, code } = refusal(error);
        this.whatsapp.status = 'failed';
        this.whatsapp.error = message;
        this.whatsapp.errorCode = code;
      } finally {
        this.whatsapp.busy = false;
      }
    },

    /**
     * Poll until the school replies, the challenge dies, or its window closes.
     *
     * The expiry is checked here rather than trusted to the server so the page stops
     * asking on its own: a tab left open overnight would otherwise poll identity every
     * two seconds until it was closed.
     */
    beginWhatsAppPolling() {
      stopPolling();
      pollTimer = setInterval(async () => {
        if (!this.whatsapp.pollSecret) {
          stopPolling();
          return;
        }
        if (this.whatsapp.expiresAt && Date.parse(this.whatsapp.expiresAt) <= Date.now()) {
          stopPolling();
          this.whatsapp.status = 'failed';
          this.whatsapp.errorCode = 'expired';
          this.whatsapp.error = 'This sign-in request has expired.';
          return;
        }

        try {
          const { data } = await identityApi.post('/v1/auth/whatsapp/status', {
            poll_secret: this.whatsapp.pollSecret,
          });
          this.whatsapp.displayName = data.display_name || '';

          if (data.status === 'code_sent') {
            stopPolling();
            this.whatsapp.status = 'code_sent';
          } else if (data.status === 'rejected') {
            stopPolling();
            this.whatsapp.status = 'failed';
            this.whatsapp.errorCode = 'rejected';
            this.whatsapp.error =
              'That number is not registered with the school, or the request expired.';
          }
        } catch {
          // A blip while polling is not a failure. The parent is mid-flow, the challenge
          // is still alive on the server, and the next tick two seconds from now is a
          // better answer than tearing their screen down over one dropped request.
        }
      }, POLL_INTERVAL_MS);
    },

    /** Submit the six digits and, on success, become signed in. */
    async submitWhatsAppCode() {
      if (this.whatsapp.busy) return;
      const code = this.whatsapp.code.trim();
      if (!code) return;

      this.whatsapp.busy = true;
      this.whatsapp.error = '';
      this.whatsapp.errorCode = '';
      try {
        const { data } = await identityApi.post('/v1/auth/whatsapp/verify', {
          poll_secret: this.whatsapp.pollSecret,
          code,
        });
        stopPolling();
        this.applySession(data);
        this.resetWhatsApp();
      } catch (error: any) {
        const { message, code: reason } = refusal(error);
        this.whatsapp.error = message;
        this.whatsapp.errorCode = reason;
        // A wrong code leaves the challenge alive and the parent on the same screen —
        // they have four more tries. Everything else has killed it, so send them back to
        // the start rather than leaving them typing into something that cannot succeed.
        if (reason && reason !== 'bad_code') {
          this.whatsapp.status = 'failed';
        }
        this.whatsapp.code = '';
      } finally {
        this.whatsapp.busy = false;
      }
    },

    /** Abandon the current attempt. Called on cancel and when the panel unmounts. */
    resetWhatsApp() {
      stopPolling();
      this.whatsapp.status = 'idle';
      this.whatsapp.pollSecret = '';
      this.whatsapp.link = '';
      this.whatsapp.message = '';
      this.whatsapp.businessNumber = '';
      this.whatsapp.expiresAt = '';
      this.whatsapp.displayName = '';
      this.whatsapp.code = '';
      this.whatsapp.error = '';
      this.whatsapp.errorCode = '';
      this.whatsapp.busy = false;
    },

    applySession(data: any) {
      this.applyTokens({
        accessToken: String(data.access_token || ''),
        refreshToken: String(data.refresh_token || ''),
      });
      this.currentUser = {
        username: data.username,
        role: data.role as UserRole,
        guardianId: data.guardian_id ?? null,
        displayName: data.display_name || '',
      };
    },

    /** Both tokens together, in memory and in storage. Rotation replaces the pair. */
    applyTokens(tokens: SessionTokens) {
      this.token = tokens.accessToken;
      this.refreshToken = tokens.refreshToken;
      writeStoredTokens(tokens);
    },

    /**
     * The session is over and it was not the parent's doing: identity refused to renew it.
     * Cleared locally — the refresh token is already dead, so there is nothing to revoke —
     * and announced, so the app can show the sign-in screen and say why.
     */
    endSession() {
      stopPolling();
      this.token = '';
      this.refreshToken = '';
      this.currentUser = null;
      clearStoredTokens();
      window.dispatchEvent(new CustomEvent('unauthorized'));
    },

    async handleLogout() {
      stopPolling();
      // Tell identity to revoke the session — every refresh token rotated from this
      // sign-in. Best effort: the local session is cleared either way, because a user
      // who clicked "log out" must end up logged out even if the network call fails.
      const refreshToken = this.refreshToken;
      this.token = '';
      this.refreshToken = '';
      this.currentUser = null;
      clearStoredTokens();

      if (refreshToken) {
        try {
          await identityApi.post('/v1/auth/logout', { refresh_token: refreshToken });
        } catch {
          // Already gone, or unreachable. Nothing useful to do.
        }
      }
    },
  },
});
