import assert from 'node:assert/strict';
import test from 'node:test';
import { VENUES } from '../src/courts';
import { getPodPlayAvailability } from '../src/podplay';
import { PREFERENCES_VERSION, restoreSelectedVenues } from '../src/preferences';

test('Gotham branches fetch their own courts, include both Hunters Point pods, and cannot mix cached events', async t => {
  const lic = VENUES.find(venue => venue.id === 'gotham')!;
  const hunters = VENUES.find(venue => venue.id === 'gotham-hunters')!;
  // Pod IDs from Gotham’s public area directory, verified October 2026.
  const licPod = { id: 'e6e67f0a-de3f-4e6e-adf5-3db126cd5c83', slug: 'courts', sport: 'PICKLEBALL' };
  const indoor = { id: 'cdaeddb7-a625-43ad-9197-9b6377528a44', slug: 'gotham-pickle-hunter-indoor-courts', sport: 'PICKLEBALL' };
  const outdoor = { id: 'a807faf4-6263-4dfa-bdc7-8d3a39c3c390', slug: 'gotham-pickle-hunter-outdoor-courts', sport: 'PICKLEBALL' };
  const unrelated = { id: 'unrelated-pod', slug: 'courts', sport: 'PICKLEBALL' };
  const otherSport = { id: 'other-sport', slug: 'other-sport', sport: 'TENNIS' };
  const event = (id: string, pod: typeof licPod) => ({
    id, name: 'Intermediate Open Play', subtype: 'OPEN_PLAY',
    startTime: '2099-10-17T23:00:00Z', endTime: '2099-10-18T01:00:00Z',
    pods: { items: [pod] }, totalTeams: 12, signups: { _total: 4 },
    admission: 'OPEN', visibility: 'LISTED', status: 'PUBLISHED',
  });
  const calls: URL[] = [];
  t.mock.method(globalThis, 'fetch', async (input: string | URL) => {
    const url = new URL(String(input));
    calls.push(url);
    if (url.hostname === 'identitytoolkit.googleapis.com') return Response.json({ idToken: 'test-guest', refreshToken: 'test-refresh', expiresIn: '3600' });
    if (url.pathname === '/apis/v2/areas') return Response.json({ items: [
      { id: 'lic', slug: 'long-island-city', pods: { items: [licPod] } },
      { id: 'hunters', slug: 'hunters-point', pods: { items: [indoor, outdoor, otherSport] } },
      { id: 'other-branch', slug: 'new-haven', pods: { items: [unrelated] } },
    ] });
    if (url.pathname === '/apis/v2/events') {
      // Deliberately overbroad results: the client must independently enforce branch scope.
      return Response.json({ items: [event('lic-open', licPod), event('indoor-open', indoor),
        event('outdoor-open', outdoor), event('other-branch-open', unrelated),
        event('other-sport-open', otherSport), event('indoor-open', indoor)] });
    }
    throw new Error(`Unexpected request: ${url.pathname}`);
  });

  const [licResult, huntersResult] = await Promise.all([
    getPodPlayAvailability(lic, '2099-10-17'), getPodPlayAvailability(hunters, '2099-10-17'),
  ]);
  assert.equal(licResult.status, 'available');
  assert.equal(huntersResult.status, 'available');
  assert.deepEqual(licResult.slots.map(slot => slot.id), ['gotham:lic-open']);
  assert.deepEqual(huntersResult.slots.map(slot => slot.id), ['gotham-hunters:indoor-open', 'gotham-hunters:outdoor-open']);
  assert.ok(huntersResult.slots.every(slot => slot.venueId === hunters.id));
  const queries = calls.filter(url => url.pathname === '/apis/v2/events').map(url => url.searchParams.getAll('podId'));
  assert.ok(queries.some(ids => ids.length === 1 && ids[0] === licPod.id));
  assert.ok(queries.some(ids => ids.length === 2 && ids.includes(indoor.id) && ids.includes(outdoor.id)));
  assert.equal(calls.filter(url => url.pathname === '/apis/v2/areas').length, 1);
  const requestCount = calls.length;
  assert.deepEqual(await getPodPlayAvailability(hunters, '2099-10-17'), huntersResult);
  assert.deepEqual(await getPodPlayAvailability(lic, '2099-10-17'), licResult);
  assert.equal(calls.length, requestCount);
});

test('returning Gotham followers see the new branch once; later deselections remain saved', () => {
  assert.deepEqual(restoreSelectedVenues({ selected: ['gotham', 'pbclub'] }), ['gotham', 'pbclub', 'gotham-hunters']);
  assert.deepEqual(restoreSelectedVenues({ selected: ['gotham'], version: PREFERENCES_VERSION }), ['gotham']);
  assert.deepEqual(restoreSelectedVenues({ selected: [] }), []);
  assert.deepEqual(restoreSelectedVenues({ selected: ['pbclub', 'removed-venue'] }), ['pbclub']);
  assert.equal(restoreSelectedVenues({}).length, 6);
});
