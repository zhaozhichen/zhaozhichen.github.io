import publicConfig from './public-config.json';
import { addDays, localDate, TIMEZONE, type Availability, type Slot, type Venue } from './courts';

// Public venue domains only. Use an independent anonymous visitor session;
// never read or forward the visitor's existing booking-account credentials.
const HOSTS = new Set(['citypickle', 'gotham', 'pbclub', 'pickleballkingdom']);
const FETCH_TIMEOUT = 15_000;
const CACHE_MS = 90_000;

type Collection<T> = { items: T[]; _total?: number; _pagination?: { count?: number; ipp?: number } };
type Pod = { id: string; slug: string; sport?: string; timezone?: string };
type Area = { id: string; slug: string; pods: Collection<Pod> };
type PublicEvent = {
  id: string; name: string; description?: string; subtype: string; startTime: string; endTime: string;
  pods: Collection<Pod>; totalTeams: number; teamSize?: number;
  signups?: { _total: number }; teams?: { _total: number } | null;
  admission?: string; admissionDate?: { regular?: string | null };
  admissionRate?: { regular?: number | null };
  admissionRating?: { minRating?: number; maxRating?: number; providerLabel?: string; connectedOnly?: boolean; duprConnectedOnly?: boolean };
  tags?: string[]; status?: string; visibility?: string; isCanceled?: boolean;
  isCustomerTagRestricted?: boolean;
};
type Guest = { key: string; token: string; refreshToken: string; expiresAt: number; areas: Area[]; areasLoadedAt: number };
type FirebaseToken = { idToken?: string; refreshToken?: string; expiresIn?: string; id_token?: string; refresh_token?: string; expires_in?: string };
type PublicSlot = Slot & { opensAt?: string; availabilityReason?: 'full' | 'not_open' | 'members_only' | 'closed' | 'restricted'; registrationUnit?: 'team' };

const guests = new Map<string, Guest>();
const guestTasks = new Map<string, Promise<Guest>>();
const cache = new Map<string, { expiresAt: number; value: Availability }>();
const requests = new Map<string, Promise<Availability>>();

class UpstreamError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

async function request(url: string, init: RequestInit = {}) {
  const response = await fetch(url, { ...init, cache: 'no-store', signal: AbortSignal.timeout(FETCH_TIMEOUT) });
  if (!response.ok) throw new UpstreamError(response.status, `Source returned ${response.status}`);
  return response;
}

function origin(host: string) {
  if (!HOSTS.has(host)) throw new Error('Unsupported venue');
  return `https://${host}.podplay.app`;
}

async function getGuest(host: string): Promise<Guest> {
  const existing = guests.get(host);
  if (existing && existing.expiresAt > Date.now() && existing.areasLoadedAt > Date.now() - 6 * 60 * 60_000) return existing;
  const active = guestTasks.get(host);
  if (active) return active;
  const task = (async () => {
    const base = origin(host);
    let guest = existing;
    if (!guest) {
      const key = publicConfig[host as keyof typeof publicConfig]?.apiKey;
      if (!key) throw new Error('Guest configuration unavailable');
      // PodPlay's own useEnsureAuth hook invokes Firebase signInAnonymously for
      // visitors. This creates the same guest session; it is not a booking.
      const auth = await (await request(`https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=${encodeURIComponent(key)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ returnSecureToken: true }),
      })).json() as FirebaseToken;
      if (!auth.idToken || !auth.refreshToken) throw new Error('Guest authentication unavailable');
      guest = { key, token: auth.idToken, refreshToken: auth.refreshToken, expiresAt: Date.now() + (Number(auth.expiresIn || 3600) - 120) * 1000, areas: [], areasLoadedAt: 0 };
      guests.set(host, guest);
    } else if (guest.expiresAt <= Date.now()) {
      const auth = await (await request(`https://securetoken.googleapis.com/v1/token?key=${encodeURIComponent(guest.key)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ grant_type: 'refresh_token', refresh_token: guest.refreshToken }).toString(),
      })).json() as FirebaseToken;
      if (!auth.id_token || !auth.refresh_token) throw new Error('Guest session expired');
      guest.token = auth.id_token;
      guest.refreshToken = auth.refresh_token;
      guest.expiresAt = Date.now() + (Number(auth.expires_in || 3600) - 120) * 1000;
    }
    if (!guest.areasLoadedAt || guest.areasLoadedAt < Date.now() - 6 * 60 * 60_000) {
      const data = await (await request(`${base}/apis/v2/areas`, { headers: { Authorization: `Bearer ${guest.token}` } })).json() as Collection<Area>;
      if (!Array.isArray(data.items)) throw new Error('Venue directory format changed');
      guest.areas = data.items;
      guest.areasLoadedAt = Date.now();
    }
    return guest;
  })();
  guestTasks.set(host, task);
  try { return await task; } finally { guestTasks.delete(host); }
}

/** Midnight in the venue's timezone, including the 23/25-hour DST days. */
export function newYorkMidnight(date: string): Date {
  const utc = Date.parse(`${date}T00:00:00Z`);
  let instant = utc;
  const formatter = new Intl.DateTimeFormat('en-US', { timeZone: TIMEZONE, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' });
  for (let i = 0; i < 3; i++) {
    const parts = Object.fromEntries(formatter.formatToParts(new Date(instant)).map(part => [part.type, part.value]));
    const represented = Date.UTC(+parts.year, +parts.month - 1, +parts.day, +parts.hour, +parts.minute, +parts.second);
    instant += utc - represented;
  }
  return new Date(instant);
}

function time(iso: string) {
  return new Intl.DateTimeFormat('en-GB', { timeZone: TIMEZONE, hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(iso));
}

function level(event: PublicEvent): string | undefined {
  const rating = event.admissionRating;
  if (rating && (typeof rating.minRating === 'number' || typeof rating.maxRating === 'number')) {
    const min = rating.minRating, max = rating.maxRating;
    return `${rating.providerLabel || 'DUPR'} ${min !== undefined ? min.toFixed(2) : ''}${min !== undefined && max !== undefined ? '–' : ''}${max !== undefined ? max.toFixed(2) : '+'}${min === undefined ? ' 以下' : ''}`;
  }
  const title = event.name.match(/(?:high|low|advanced)\s+(?:intermediate|beginner)|intermediate|beginner|advanced|all levels/i)?.[0];
  return title || event.tags?.find(tag => /beginner|intermediate|advanced|expert|all levels|\d\.\d/i.test(tag));
}

function words(text: string) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function isNonPlayingEntry(event: PublicEvent): boolean {
  const title = words(event.name), description = words(event.description || '');
  const tags = (event.tags || []).map(words);
  // Venues sometimes label spectator tickets OPEN_PLAY. Require explicit
  // nonplaying language, so "spectators welcome" on a player event is retained.
  return /\b(?:spectator|observer|audience|non player|non playing) (?:ticket|pass|registration|admission|only)\b/.test(title)
    || /\bvolunteer (?:registration|sign up|signup|shift)\b/.test(title)
    || /\b(?:spectator|observer|audience|non player|non playing)(?: (?:ticket|pass|admission))? only\b/.test(description)
    || tags.some(tag => /^(?:spectator|spectators|spectator ticket|volunteer registration|non playing)$/.test(tag));
}

function isInvitationRestricted(event: PublicEvent): boolean {
  const title = words(event.name), description = words(event.description || '');
  const invitationOnly = /\b(?:invite only|invitation only|invited guests only|do not register unless invited)\b/;
  return invitationOnly.test(title) || invitationOnly.test(description)
    || /\bprivate (?:game|event|court contract)\b/.test(title)
    || (event.tags || []).some(tag => /^(?:private event|private game|invite only|invitation only)$/.test(words(tag)));
}

export function publicEventToSlot(event: PublicEvent, venue: Venue, date: string, now = Date.now()): PublicSlot | null {
  if (event.subtype !== 'OPEN_PLAY' || event.isCanceled || event.status !== 'PUBLISHED' || event.visibility !== 'LISTED') return null;
  if (isNonPlayingEntry(event)) return null;
  const starts = Date.parse(event.startTime), ends = Date.parse(event.endTime);
  if (!Number.isFinite(starts) || !Number.isFinite(ends) || ends <= starts || localDate(new Date(starts)) !== date) return null;
  const isTeam = (event.teamSize || 1) > 1;
  const used = isTeam ? event.teams?._total : event.signups?._total;
  const remaining = Number.isFinite(event.totalTeams) && typeof used === 'number' ? Math.max(0, event.totalTeams - used) : undefined;
  const opensAt = event.admissionDate?.regular || undefined;
  let availabilityReason: PublicSlot['availabilityReason'];
  if (starts <= now || event.admission === 'CLOSED') availabilityReason = 'closed';
  else if (event.isCustomerTagRestricted || isInvitationRestricted(event)) availabilityReason = 'restricted';
  else if (event.admission === 'MEMBERS_ONLY') availabilityReason = 'members_only';
  // Missing or new upstream admission values must never imply public access.
  else if (event.admission !== 'OPEN') availabilityReason = 'restricted';
  else if (remaining === 0) availabilityReason = 'full';
  else if (opensAt && Date.parse(opensAt) > now) availabilityReason = 'not_open';
  const end = localDate(new Date(ends)) === date ? time(event.endTime) : `${Number(time(event.endTime).slice(0, 2)) + 24}${time(event.endTime).slice(2)}`;
  return {
    id: `${venue.id}:${event.id}`, venueId: venue.id, date, start: time(event.startTime), end,
    duration: Math.round((ends - starts) / 60_000), bookingUrl: `${origin(venue.host!)}/community/events/${encodeURIComponent(event.id)}`,
    kind: 'open-play', title: event.name.trim(), remaining, capacity: event.totalTeams,
    price: typeof event.admissionRate?.regular === 'number' ? event.admissionRate.regular : undefined,
    currency: 'USD', level: level(event), available: !availabilityReason && remaining !== undefined && remaining > 0,
    opensAt, availabilityReason, ...(isTeam ? { registrationUnit: 'team' as const } : {}),
  };
}

async function load(venue: Venue, date: string): Promise<Availability> {
  const result = (status: Availability['status'], slots: Slot[] = [], message?: string): Availability => ({ venueId: venue.id, date, checkedAt: new Date().toISOString(), status, slots, message });
  try {
    const host = venue.host!;
    const guest = await getGuest(host);
    const area = guest.areas.find(item => item.slug === venue.slug);
    if (!area) return result('not_listed', [], '官方目录暂未列出此分馆');
    const pods = area.pods.items.filter(pod => pod.sport === 'PICKLEBALL' && (!venue.pod || pod.slug === venue.pod));
    if (!pods.length) return result('not_listed', [], '官方目录暂未列出匹克球活动');
    const podIds = new Set(pods.map(pod => pod.id));
    const query = new URLSearchParams({ selfOnly: 'true', excludeClosedSeries: 'true', excludeUnlisted: 'true', includeCanceledInvitations: 'false', startTime: newYorkMidnight(date).toISOString(), endTime: new Date(newYorkMidnight(addDays(date, 1)).getTime() - 1).toISOString(), sort: 'startTime', ipp: '100' });
    for (const pod of pods) query.append('podId', pod.id);
    const slots = new Map<string, Slot>();
    // Match the public listing's pagination. Never request attendees/signups.
    for (let page = 1; page <= 10; page++) {
      query.set('page', String(page));
      const data = await (await request(`${origin(host)}/apis/v2/events?${query}`, { headers: { Authorization: `Bearer ${guest.token}`, 'Accept-Language': 'en' } })).json() as Collection<PublicEvent>;
      if (!Array.isArray(data.items)) throw new Error('Event response format changed');
      for (const event of data.items) {
        if (!event.pods?.items?.some(pod => podIds.has(pod.id))) continue;
        const slot = publicEventToSlot(event, venue, date);
        if (slot) slots.set(slot.id, slot);
      }
      if (data.items.length < 100) break;
      if (page === 10) throw new Error('Event list exceeds supported page limit');
    }
    const all = [...slots.values()].sort((a, b) => a.start.localeCompare(b.start));
    return result(all.length ? 'available' : 'empty', all, all.length ? undefined : '官方当前未列出当天的 Open Play；不代表场馆全天已满');
  } catch (error) {
    const needsLogin = error instanceof UpstreamError && [401, 403].includes(error.status);
    return result(needsLogin ? 'login_required' : 'unavailable', [], needsLogin ? '官方访客接口暂不可用，请前往官方查看' : '暂时无法读取官方数据，请稍后刷新或前往官方查看');
  }
}

export async function getPodPlayAvailability(venue: Venue, date: string, refresh = false): Promise<Availability> {
  const key = `${venue.id}:${date}`;
  const saved = cache.get(key);
  if (saved && saved.expiresAt > Date.now() && (!refresh || Date.parse(saved.value.checkedAt) > Date.now() - 10_000)) return saved.value;
  const active = requests.get(key);
  if (active) return active;
  const task = load(venue, date);
  requests.set(key, task);
  try {
    const value = await task;
    if (cache.size >= 256) cache.delete(cache.keys().next().value!);
    cache.set(key, { value, expiresAt: Date.now() + (value.status === 'unavailable' || value.status === 'login_required' ? 20_000 : CACHE_MS) });
    return value;
  } finally { requests.delete(key); }
}
