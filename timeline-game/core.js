/* Shared by the browser and the rule tests; no DOM or storage side effects. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TimelineGame = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const MIN_ANSWERS = 10;
  function shuffle(ids, random = Math.random) {
    const result = [...ids];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
  function fromOrder(name, order, id, startedAt) {
    return {version: 1, id, name: name.trim().slice(0, 24), order, cursor: 1,
      timeline: [order[0]], phase: 'guess', attempts: [], result: null,
      appeared: 2, startedAt, endedAt: null};
  }
  function create(name, cards, random = Math.random, id = String(Date.now()), now = Date.now()) {
    if (!name.trim() || cards.length < 2) throw new Error('需要玩家名字与至少两张卡片。');
    return fromOrder(name, shuffle(cards.map(c => c.id), random), id, now);
  }
  function pending(state) { return state.order[state.cursor]; }
  function place(state, slot, byId) {
    if (state.phase !== 'guess' || !Number.isInteger(slot) || slot < 0 || slot > state.timeline.length) return state;
    const id = pending(state), year = byId[id].year;
    const lowerYear = slot ? byId[state.timeline[slot - 1]].year : null;
    const upperYear = slot < state.timeline.length ? byId[state.timeline[slot]].year : null;
    const correct = (lowerYear === null || lowerYear <= year) && (upperYear === null || year <= upperYear);
    let actualSlot = correct ? slot : state.timeline.findIndex(key => byId[key].year > year);
    if (actualSlot < 0) actualSlot = state.timeline.length;
    const timeline = [...state.timeline]; timeline.splice(actualSlot, 0, id);
    const result = {cardId: id, chosenSlot: slot, correct, actualSlot, lowerYear, upperYear};
    return {...state, phase: 'reveal', timeline, result,
      attempts: [...state.attempts, {id, slot, correct}]};
  }
  function next(state, now = Date.now()) {
    if (state.phase !== 'reveal') return state;
    if (state.cursor + 1 >= state.order.length) return finish(state, now);
    return {...state, cursor: state.cursor + 1, appeared: state.appeared + 1, phase: 'guess', result: null};
  }
  function finish(state, now = Date.now()) {
    if (state.phase === 'finished' || !canFinish(state)) return state;
    return {...state, finishedFrom: state.phase, phase: 'finished', endedAt: now};
  }
  function canFinish(state) {
    return state && state.phase !== 'finished' && state.attempts.length >= Math.min(MIN_ANSWERS, state.order.length - 1);
  }
  function stats(state) {
    const answered = state.attempts.length, correct = state.attempts.filter(a => a.correct).length;
    return {appeared: state.appeared, answered, correct, wrong: answered - correct,
      accuracy: answered ? correct / answered : null, remaining: state.order.length - state.appeared};
  }
  // Replay saved choices so counts, positions and scores cannot drift after a refresh.
  function restore(saved, cards) {
    try {
      const byId = Object.fromEntries(cards.map(c => [c.id, c]));
      if (!saved || saved.version !== 1 || typeof saved.name !== 'string' || !saved.name.trim() ||
          typeof saved.id !== 'string' || !Number.isFinite(saved.startedAt) ||
          !Array.isArray(saved.order) || saved.order.length !== cards.length ||
          new Set(saved.order).size !== cards.length || saved.order.some(id => !byId[id]) ||
          !Array.isArray(saved.attempts) || saved.attempts.length >= cards.length ||
          !['guess','reveal','finished'].includes(saved.phase)) return null;
      let state = fromOrder(saved.name, [...saved.order], saved.id, saved.startedAt);
      const target = saved.phase === 'finished' ? saved.finishedFrom : saved.phase;
      if (!['guess', 'reveal'].includes(target) || (target === 'reveal' && !saved.attempts.length)) return null;
      for (let i = 0; i < saved.attempts.length; i++) {
        const a = saved.attempts[i];
        if (a.id !== pending(state)) return null;
        const placed = place(state, a.slot, byId);
        if (placed === state) return null;
        state = placed;
        if (i < saved.attempts.length - 1 || target === 'guess') state = next(state);
      }
      if (state.phase !== target) return null;
      if (saved.phase === 'finished') {
        if (!Number.isFinite(saved.endedAt)) return null;
        // An old short round resumes instead of bypassing the ten-answer rule.
        state = finish(state, saved.endedAt);
      }
      return state;
    } catch { return null; }
  }
  return {MIN_ANSWERS, shuffle, create, pending, place, next, finish, canFinish, stats, restore};
});
