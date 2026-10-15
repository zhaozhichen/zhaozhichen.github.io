import type { Availability, Slot } from './courts';
import { localDate, minutes, TIMEZONE } from './courts';

const API = 'https://api.roller.app/api/checkout/onlinecheckout/product-availability';
const CHECKOUT = 'https://ecom.roller.app/gameonnewjersey/onlinecheckout/en-us';
const OPEN_PLAY_PRODUCTS = new Set(['19981', '20753', '20739']);

type RollerTicket = {
  id: string;
  cost?: number;
  isSuspended?: boolean;
};
type RollerAllocation = {
  productId: string;
  capacityRemaining?: number | null;
  bookableCapacityRemaining?: number | null;
};
type RollerSession = {
  date: string;
  startTime: string;
  endTime: string;
  onlineSalesOpen: boolean;
  ticketCapacityRemaining?: number | null;
  capacityRemaining?: number | null;
  allocations?: RollerAllocation[];
};
type RollerProduct = {
  id: string;
  name: string;
  description?: string;
  type: string;
  products?: RollerTicket[];
  sessions?: RollerSession[];
};

function capacity(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? Math.floor(value)
    : undefined;
}

/** Only public schedule/product data. This endpoint never creates a cart or reservation. */
export async function getGameOnAvailability(date: string): Promise<Availability> {
  const now = new Date();
  const base = { venueId: 'gameon', date, checkedAt: now.toISOString(), slots: [] as Slot[] };
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return { ...base, status: 'unavailable', message: '日期格式无效。' };
  }

  try {
    const response = await fetch(`${API}?date=${encodeURIComponent(date)}`, {
      // These are the public venue/checkout identifiers sent by the official site,
      // not an account credential. No guest data or authenticated endpoints are used.
      headers: {
        Accept: 'application/json',
        'X-Api-Key': 'gameonnewjersey',
        'X-Cell-Id': 'j',
        'X-Checkout-Slug': 'onlinecheckout',
        'X-Origin-Id': '1',
      },
      signal: AbortSignal.timeout(12_000),
      cache: 'no-store',
    });
    if (!response.ok) throw new Error(`Roller ${response.status}`);
    const data: unknown = await response.json();
    if (!Array.isArray(data)) throw new Error('Invalid Roller response');

    const today = localDate(now);
    const currentTime = new Intl.DateTimeFormat('en-GB', {
      timeZone: TIMEZONE, hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
    }).format(now);
    const slots: Slot[] = [];
    const seen = new Set<string>();
    let listed = false;

    for (const raw of data) {
      if (!raw || typeof raw !== 'object') continue;
      const product = raw as RollerProduct;
      // The venue lists these three as pickleball Open Play. Membership products
      // and private court rentals must never become calendar sessions.
      const isOpenPlay = OPEN_PLAY_PRODUCTS.has(String(product.id))
        || (/pickleball/i.test(product.description ?? '') && /open[ -]?play/i.test(product.name ?? ''));
      if (!isOpenPlay || product.type !== 'sessionpass') continue;
      listed = true;
      if (!Array.isArray(product.sessions) || !Array.isArray(product.products)) continue;
      const tickets = product.products.filter(ticket => ticket && !ticket.isSuspended);

      for (const session of product.sessions) {
        if (!session || session.date !== date
          || !/^\d{2}:\d{2}$/.test(session.startTime)
          || !/^\d{2}:\d{2}$/.test(session.endTime)) continue;
        const start = minutes(session.startTime);
        const end = minutes(session.endTime);
        if (start < 0 || start >= 1440 || end <= start || end > 1440) continue;
        const id = `gameon-${product.id}-${date}-${session.startTime}`;
        if (seen.has(id)) continue;
        seen.add(id);

        const allocations = (session.allocations ?? []).filter(allocation =>
          tickets.some(ticket => String(ticket.id) === String(allocation.productId)));
        const bookable = allocations.filter(allocation => {
          const remaining = capacity(allocation.bookableCapacityRemaining)
            ?? capacity(allocation.capacityRemaining);
          return remaining === undefined || remaining > 0;
        });
        const remaining = capacity(session.ticketCapacityRemaining)
          ?? capacity(session.capacityRemaining);
        const stillUpcoming = date > today || (date === today && start > minutes(currentTime));
        const available = session.onlineSalesOpen === true && stillUpcoming
          && bookable.length > 0 && (remaining === undefined || remaining > 0);
        const prices = tickets
          .filter(ticket => allocations.some(allocation => String(allocation.productId) === String(ticket.id)))
          .map(ticket => ticket.cost)
          .filter((cost): cost is number => typeof cost === 'number' && Number.isFinite(cost) && cost >= 0);

        slots.push({
          id, venueId: 'gameon', date,
          start: session.startTime, end: session.endTime, duration: end - start,
          kind: 'open-play', title: product.name, remaining,
          price: prices.length ? Math.min(...prices) : undefined,
          currency: 'USD', available,
          bookingUrl: `${CHECKOUT}/product/${encodeURIComponent(product.id)}?date=${date}`,
        });
      }
    }

    slots.sort((left, right) => left.start.localeCompare(right.start));
    return {
      ...base,
      slots,
      status: slots.some(slot => slot.available) ? 'available' : listed ? 'empty' : 'not_listed',
      message: slots.length
        ? '价格为单人基础票价，税费另计；会员优惠及免预约规则以场馆说明为准。'
        : '官方页面未列出当天可预约的 Open Play；不代表场馆全天没有活动。',
    };
  } catch {
    return { ...base, status: 'unavailable', message: '暂时无法读取 Game On 的公开场次，请打开官方页面确认。' };
  }
}
