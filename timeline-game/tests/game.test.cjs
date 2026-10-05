const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const G = require('../core.js');
const deck = JSON.parse(fs.readFileSync(path.join(__dirname, '../cards/cards.json'))).cards;
const mini = [{id:'a',year:100},{id:'b',year:50},{id:'c',year:100},{id:'d',year:200},{id:'e',year:-200}];
const map = Object.fromEntries(mini.map(c => [c.id,c]));
const fixed = () => G.create('玩家', mini, () => .999, 'test-game', 10);

test('a round begins with one known card, one pending card, and no scored answer', () => {
  const s = fixed();
  assert.deepEqual(s.timeline, ['a']); assert.equal(G.pending(s),'b');
  assert.deepEqual(G.stats(s), {appeared:2,answered:0,correct:0,wrong:0,accuracy:null,remaining:3});
});
test('left and right extremes accept dates outside all known years, including BCE', () => {
  let s = G.place(fixed(), 0, map);
  assert.equal(s.result.correct, true); assert.deepEqual(s.timeline,['b','a']);
  s = G.next(s); s = G.place(s, 2, map); s = G.next(s); s = G.place(s, 3, map);
  assert.equal(s.result.correct, true); s = G.next(s); s = G.place(s, 0, map);
  assert.equal(s.result.correct, true); assert.deepEqual(s.timeline,['e','b','a','c','d']);
});
test('a wrong answer is counted once and is inserted at its actual sorted position', () => {
  const s = G.place(fixed(), 1, map);
  assert.equal(s.result.correct,false); assert.deepEqual(s.timeline,['b','a']);
  assert.equal(G.stats(s).wrong,1); assert.equal(G.place(s,0,map),s);
});
test('all slots adjacent to an equal-year group are accepted', () => {
  const cards = [{id:'a',year:100},{id:'b',year:100},{id:'c',year:100}];
  const lookup = Object.fromEntries(cards.map(c => [c.id,c]));
  let s = G.create('同年',cards,()=>.999);
  for (const slot of [0,1]) assert.equal(G.place(s,slot,lookup).result.correct,true);
  s = G.next(G.place(s,1,lookup));
  for (const slot of [0,1,2]) assert.equal(G.place(s,slot,lookup).result.correct,true);
});
test('middle slots are judged by both neighbors', () => {
  const cards = [{id:'a',year:100},{id:'b',year:300},{id:'c',year:200}];
  const lookup = Object.fromEntries(cards.map(c => [c.id,c]));
  let s = G.create('中间',cards,()=>.999); s=G.next(G.place(s,1,lookup));
  assert.equal(G.place(s,1,lookup).result.correct,true);
  assert.equal(G.place(s,0,lookup).result.correct,false);
  assert.equal(G.place(s,2,lookup).result.correct,false);
});
test('invalid slots and repeated next actions cannot change the score', () => {
  const s = fixed();
  for (const slot of [-1,2,1.5,NaN]) assert.equal(G.place(s,slot,map),s);
  assert.equal(G.next(s),s);
});
test('accuracy excludes the anchor and unanswered current card', () => {
  let s = G.next(G.place(fixed(),1,map)); s = G.next(G.place(s,2,map));
  assert.deepEqual(G.stats(s),{appeared:4,answered:2,correct:1,wrong:1,accuracy:.5,remaining:1});
  s=G.finish(s,30); assert.equal(G.stats(s).accuracy,.5);
});
test('the full real deck deals without duplicates and ends after exactly 283 answers', () => {
  const lookup=Object.fromEntries(deck.map(c=>[c.id,c]));
  let s=G.create('全卡测试',deck,()=>.42,'all',1);
  assert.equal(new Set(s.order).size,284);
  while(s.phase!=='finished') {
    const year=lookup[G.pending(s)].year;
    const slot=s.timeline.filter(id=>lookup[id].year<=year).length;
    s=G.place(s,slot,lookup); assert.equal(s.result.correct,true); s=G.next(s,1000);
  }
  assert.deepEqual(G.stats(s),{appeared:284,answered:283,correct:283,wrong:0,accuracy:1,remaining:0});
  assert.equal(s.timeline.length,284); assert.equal(G.next(s),s);
  assert.deepEqual(G.restore(s,deck),s);
});
test('a wrong placement still preserves sorted order over an entire deck', () => {
  const lookup=Object.fromEntries(deck.map(c=>[c.id,c])); let s=G.create('任意位置',deck,()=>.73);
  while(s.phase!=='finished') {
    s=G.place(s,Math.floor(s.timeline.length/2),lookup);
    const years=s.timeline.map(id=>lookup[id].year);
    assert.deepEqual(years,[...years].sort((a,b)=>a-b)); s=G.next(s);
  }
  assert.equal(G.stats(s).answered,283);
});
test('guess, reveal and finished games restore by replay, with exactly the same scores', () => {
  let s = fixed(); const states = [s, G.finish(s,20)];
  s=G.place(s,1,map); states.push(s,G.finish(s,25));
  s=G.next(s); states.push(s,G.finish(s,30));
  s=G.place(s,2,map); states.push(s);
  for(const state of states) assert.deepEqual(G.restore(JSON.parse(JSON.stringify(state)),mini),state);
});
test('damaged saves are rejected and stale derived counts are recomputed', () => {
  const s=G.place(fixed(),1,map);
  assert.equal(G.restore({...s,order:['a','a']},mini),null);
  assert.equal(G.restore({...s,attempts:[{id:'e',slot:0}]},mini),null);
  assert.equal(G.restore({...s,attempts:[{id:'b',slot:900}]},mini),null);
  const restored=G.restore({...s,appeared:1000,timeline:['e'],attempts:[{id:'b',slot:1,correct:true}]},mini);
  assert.equal(G.stats(restored).wrong,1); assert.deepEqual(restored.timeline,['b','a']);
});
test('ending is disabled through nine answers and enabled at ten', () => {
  const lookup=Object.fromEntries(deck.map(c=>[c.id,c]));
  let s=G.create('十张门槛',deck,()=>.55,'min-ten',1);
  for(let i=0;i<10;i++) {
    assert.equal(G.canFinish(s),false); assert.equal(G.finish(s,20),s);
    s=G.place(s,0,lookup);
    if(i<9) { assert.equal(G.canFinish(s),false); s=G.next(s); }
  }
  assert.equal(G.stats(s).answered,10); assert.equal(G.canFinish(s),true);
  const ended=G.finish(s,42); assert.equal(ended.phase,'finished'); assert.equal(ended.endedAt,42);
  assert.deepEqual(G.restore(ended,deck),ended);
  const continuing=G.next(s); assert.equal(continuing.phase,'guess'); assert.equal(G.canFinish(continuing),true);
  assert.equal(G.stats(continuing).answered,10);
});
test('an old finished short round resumes under the ten-answer rule', () => {
  const lookup=Object.fromEntries(deck.map(c=>[c.id,c]));
  const s=G.place(G.create('原存档',deck,()=>.3,'old-short',1),0,lookup);
  const old={...s,phase:'finished',finishedFrom:'reveal',endedAt:99};
  const restored=G.restore(old,deck);
  assert.equal(restored.phase,'reveal'); assert.equal(restored.attempts.length,1);
  assert.equal(G.canFinish(restored),false);
});
test('separate rounds with the same name have independent completion times', () => {
  const lookup=Object.fromEntries(deck.map(c=>[c.id,c]));
  function round(id,end) {
    let s=G.create('同一个名字',deck,()=>.5,id,1);
    for(let i=0;i<10;i++){s=G.place(s,0,lookup);if(i<9)s=G.next(s);}
    return G.finish(s,end);
  }
  const first=round('one',100), second=round('two',200);
  assert.notEqual(first.id,second.id); assert.notEqual(first.endedAt,second.endedAt);
  assert.equal(first.name,second.name); assert.equal(first.attempts.length,10);
});

test('the share image represents every answer in order, including long rounds', async () => {
  const vm = require('node:vm');
  const source = fs.readFileSync(path.join(__dirname, '../share.js'), 'utf8');
  const lookup = Object.fromEntries(deck.map(c => [c.id, c]));
  for (const count of [10, 11, 30, 31, 100, deck.length - 1]) {
    let state = G.create('玩家', deck, () => .8, 'share-test');
    for (let i = 0; i < count; i++) {
      const year = lookup[G.pending(state)].year;
      const slot = i % 3 ? state.timeline.filter(id => lookup[id].year <= year).length : 0;
      state = G.place(state, slot, lookup);
      if (i < count - 1) state = G.next(state);
    }
    state = G.finish(state);
    const dots = [], labels = []; let circle;
    const ctx = {
      fillRect() {}, strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
      measureText() { return {width: 100}; }, fillText(value) { labels.push(value); },
      arc(x, y, r) { circle = {x, y, r}; }, fill() { dots.push({...circle, color: this.fillStyle}); }
    };
    const canvas = {getContext: () => ctx};
    const window = {TimelineGame: G, TIMELINE_QR: {matrix: [[true]]}};
    vm.runInNewContext(source, {window, document: {fonts: {load: async () => [{}]}, createElement: () => canvas}});
    await window.TimelineShare.make(state);
    assert.equal(dots.length, count, `${count} answers must produce ${count} dots`);
    assert.deepEqual(dots.map(d => d.color), state.attempts.map(a => a.correct ? '#fff' : '#000'));
    assert.ok(labels.includes(`答对 ${G.stats(state).correct} / ${count} 张`));
    for (let i = 0; i < dots.length; i++) {
      const dot = dots[i];
      assert.ok(dot.x - dot.r >= 80 && dot.x + dot.r <= 1000);
      assert.ok(dot.y - dot.r > 720 && dot.y + dot.r < 950, 'dots must stay between score and QR');
      if (i) assert.ok(dot.y > dots[i - 1].y || (dot.y === dots[i - 1].y && dot.x > dots[i - 1].x), 'read left to right, then top to bottom');
      for (const other of dots.slice(0, i)) assert.ok(Math.hypot(dot.x - other.x, dot.y - other.y) > dot.r + other.r + 2, 'dots must not overlap');
    }
  }
});
