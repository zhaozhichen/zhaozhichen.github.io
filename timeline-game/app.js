(function () {
  'use strict';
  const G = window.TimelineGame, cards = window.TIMELINE_CARDS.cards;
  const byId = Object.fromEntries(cards.map(c => [c.id, c]));
  const $ = selector => document.querySelector(selector);
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const asset = (card, side) => 'cards/' + card[side];
  const KEY = 'calculating-empires.timeline-game.v1';
  let storageWorks = true, archivePage = 0, archiveSide = 'front', archiveCards = cards;
  let loaded = {};
  function notice(message) { $('#storage-notice').textContent = message; $('#storage-notice').hidden = false; }
  try { loaded = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; }
  catch { notice('未能读取本地存档。可以开始新游戏；原有记录暂时无法恢复。'); }
  let state = G.restore(loaded.active, cards);
  let lastName = typeof loaded.lastName === 'string' ? loaded.lastName.slice(0, 24) : '';
  if (loaded.active && !state) notice('上次进度无法恢复，请开始新的一局。');
  try { const probe = KEY + '.probe'; localStorage.setItem(probe, '1'); localStorage.removeItem(probe); }
  catch { storageWorks = false; notice('此浏览器不允许本地保存。仍可玩游戏，但关闭页面后进度和新成绩将丢失。'); }
  function save() {
    if (!storageWorks) return;
    try { localStorage.setItem(KEY, JSON.stringify({version: 1, active: state, lastName})); }
    catch { storageWorks = false; notice('本地保存失败。本局可以继续，但关闭页面后新进度和新成绩可能丢失。'); }
  }
  const pct = value => value === null ? '—' : (value * 100).toFixed(1) + '%';
  const cardImage = (c, side, lazy = true) => {
    const image = make('img'); image.src = asset(c, side);
    image.alt = c.title + (side === 'back' ? '，' + c.year + ' 年，答案面' : '，事件面');
    image.width = 1008; image.height = 1408; image.decoding = 'async';
    if (lazy) image.loading = 'lazy';
    return image;
  };
  function newGame(name) {
    lastName = name;
    const id = window.crypto?.randomUUID?.() || Date.now() + '-' + Math.random().toString(36).slice(2);
    state = G.create(name, cards, Math.random, id);
    save(); renderPlay(); focusVisibleSlot();
  }
  $('#start-form').addEventListener('submit', event => {
    event.preventDefault();
    const field = $('#player-name'), name = field.value.trim();
    if (!name) { field.setCustomValidity('请输入你的名字。'); field.reportValidity(); return; }
    newGame(name);
  });
  $('#player-name').addEventListener('input', () => $('#player-name').setCustomValidity(''));

  function renderPlay() {
    $('#welcome').hidden = !!state;
    $('#game').hidden = !state || state.phase === 'finished';
    $('#finished').hidden = !state || state.phase !== 'finished';
    if (!state) { $('#player-name').value = lastName; return; }
    if (state.phase === 'finished') { renderFinished(); return; }
    const s = G.stats(state), c = byId[G.pending(state)], revealed = state.phase === 'reveal';
    $('#player-label').textContent = state.name;
    $('#end-game').disabled = !G.canFinish(state);
    $('#end-game').textContent = G.canFinish(state) ? '结束' : `结束 · ${s.answered}/10`;
    $('#end-game').setAttribute('aria-label', G.canFinish(state) ? '结束本局' : `结束需作答 10 张，已作答 ${s.answered} 张`);
    for (const key of ['answered','correct','wrong']) $('#stat-' + key).textContent = s[key];
    $('#stat-accuracy').textContent = pct(s.accuracy);
    $('#current-image').src = asset(c, revealed ? 'back' : 'front');
    $('#current-image').alt = c.title + (revealed ? '，' + c.year + ' 年，答案面' : '，请判断时间位置');
    $('#current-card').disabled = false;
    $('#current-card').setAttribute('aria-label', revealed ? '放大阅读：' + c.title : '待放置事件：' + c.title);
    $('.round-copy').hidden = !revealed;
    $('#round-title').textContent = revealed ? (state.result.correct ? '✓ 正确' : '× 错误') : '';
    $('#next-card').textContent = state.cursor === cards.length - 1 ? '完成' : '下一张';
    renderTimeline();
  }
  function slotLabel(index) {
    const left = index ? byId[state.timeline[index - 1]] : null;
    const right = index < state.timeline.length ? byId[state.timeline[index]] : null;
    return `空位 ${index + 1}：` + (left && right ? `${left.year} 年的「${left.title}」与 ${right.year} 年的「${right.title}」之间` : left ? `${left.year} 年的「${left.title}」之后` : `${right.year} 年的「${right.title}」之前`);
  }
  function renderTimeline() {
    const root = $('#timeline'), oldScroll = $('#timeline-scroll').scrollLeft;
    const nodes = document.createDocumentFragment();
    for (let i = 0; i <= state.timeline.length; i++) {
      const slot = make('button', 'slot'); slot.type = 'button'; slot.dataset.slot = i;
      slot.setAttribute('aria-label', slotLabel(i)); slot.title = slotLabel(i);
      slot.disabled = state.phase !== 'guess';
      const plus = make('span', 'plus', '+'); plus.setAttribute('aria-hidden', 'true');
      slot.append(plus);
      slot.addEventListener('click', () => answer(i)); nodes.append(slot);
      if (i < state.timeline.length) {
        const c = byId[state.timeline[i]];
        const button = make('button', 'placed-card' + (state.phase === 'reveal' && state.result.cardId === c.id ? ' recent' : ''));
        button.type = 'button'; button.dataset.card = c.id;
        button.setAttribute('aria-label', `${c.year} 年，${c.title}，查看答案`);
        button.append(cardImage(c, 'back'));
        button.addEventListener('click', () => openCard(c)); nodes.append(button);
      }
    }
    root.replaceChildren(nodes);
    $('#timeline-scroll').scrollLeft = oldScroll;
    requestAnimationFrame(syncTimelineTools);
  }
  function answer(slot) {
    if (!state || state.phase !== 'guess') return;
    state = G.place(state, slot, byId); save(); renderPlay();
    requestAnimationFrame(() => {
      const recent = $('#timeline .recent'), scroller = $('#timeline-scroll');
      if (recent) {
        const offset = recent.getBoundingClientRect().left - scroller.getBoundingClientRect().left;
        scroller.scrollTo({left: scroller.scrollLeft + offset - scroller.clientWidth / 2 + recent.clientWidth / 2, behavior: motion()});
      }
      $('#next-card').focus({preventScroll: true});
    });
  }
  $('#next-card').addEventListener('click', () => {
    if (!state || state.phase !== 'reveal') return;
    state = G.next(state);
    save(); renderPlay();
    if (state.phase === 'guess') focusVisibleSlot();
    else $('#finished-title').focus();
  });
  function focusVisibleSlot() {
    const scroll = $('#timeline-scroll'), rect = scroll.getBoundingClientRect();
    const slots = [...$('#timeline').querySelectorAll('.slot')];
    const visible = slots.find(s => { const r = s.getBoundingClientRect(); return r.left >= rect.left && r.right <= rect.right; });
    (visible || slots[0])?.focus({preventScroll: true});
  }
  function syncTimelineTools() {
    const scroller = $('#timeline-scroll'), max = scroller.scrollWidth - scroller.clientWidth;
    $('.timeline-tools').hidden = max <= 2;
    $('#timeline-range').disabled = max <= 2;
    $('#timeline-range').value = max > 0 ? Math.round(scroller.scrollLeft / max * 1000) : 0;
    $('#timeline-range').setAttribute('aria-valuetext', max > 2 ? `时间轴的 ${Math.round(scroller.scrollLeft / max * 100)}% 位置` : '所有事件均已显示');
    $('#timeline-start').disabled = $('#timeline-left').disabled = scroller.scrollLeft <= 2;
    $('#timeline-end').disabled = $('#timeline-right').disabled = max - scroller.scrollLeft <= 2;
  }
  const motion = () => matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
  $('#timeline-scroll').addEventListener('scroll', syncTimelineTools, {passive: true});
  window.addEventListener('resize', syncTimelineTools);
  $('#timeline-range').addEventListener('input', event => {
    const scroller = $('#timeline-scroll');
    scroller.scrollTo({left: event.target.value / 1000 * (scroller.scrollWidth - scroller.clientWidth), behavior: 'instant'});
  });
  for (const [id, delta] of [['left', -1], ['right', 1]]) $('#timeline-' + id).onclick = () => $('#timeline-scroll').scrollBy({left: delta * $('#timeline-scroll').clientWidth * .7, behavior: motion()});
  $('#timeline-start').onclick = () => $('#timeline-scroll').scrollTo({left: 0, behavior: motion()});
  $('#timeline-end').onclick = () => $('#timeline-scroll').scrollTo({left: $('#timeline-scroll').scrollWidth, behavior: motion()});
  $('#current-card').onclick = () => { if (state && state.phase !== 'finished') openCard(byId[G.pending(state)], state.phase === 'guess'); };
  $('#end-game').onclick = () => {
    if (!G.canFinish(state)) return;
    state = G.finish(state); save(); renderPlay(); $('#finished-title').focus();
  };
  async function renderFinished() {
    const finishedState = state, score = G.stats(finishedState), root = $('#finished');
    const summary = make('section', 'finish-summary');
    const h = make('h1', 'sr-only', '本局成绩'); h.id = 'finished-title'; h.tabIndex = -1;
    const status = make('p', 'muted', '正在生成成绩卡…'); status.setAttribute('role', 'status');
    const actions = make('div', 'finish-actions');
    const again = make('button', '', '再玩一次');
    again.onclick = () => { state = null; save(); renderPlay(); $('#player-name').focus(); };
    actions.append(again); summary.append(h, status, actions);
    const wrong = finishedState.attempts.filter(a => !a.correct);
    if (wrong.length) {
      const review = make('details', 'review-list'); review.append(make('summary', '', `错题 · ${wrong.length}`));
      const list = make('ul');
      for (const a of wrong) {
        const c = byId[a.id], li = make('li'), button = make('button');
        button.append(make('span', '', String(c.year)), make('span', '', c.title));
        button.onclick = () => openCard(c); li.append(button); list.append(li);
      }
      review.append(list); summary.append(review);
    }
    // Mount the heading before awaiting fonts so finishing can focus it immediately.
    root.replaceChildren(summary);
    const isCurrent = () => state === finishedState && summary.isConnected;
    try {
      const canvas = await window.TimelineShare.make(finishedState);
      if (!isCurrent()) return;
      const url = canvas.toDataURL('image/png');
      const image = make('img', 'share-card'); image.src = url; image.width = 1080; image.height = 1350;
      image.alt = `${finishedState.name}，正确率 ${pct(score.accuracy)}，答对 ${score.correct} / ${score.answered} 张，结束于 ${window.TimelineShare.timestamp(finishedState.endedAt)}。二维码指向 ${window.TIMELINE_QR.url}`;
      status.replaceWith(image);
      const download = make('a', 'button primary', '保存图片'); download.id = 'save-share'; download.href = url; download.download = window.TimelineShare.fileName(finishedState);
      actions.prepend(download);
      // Native sharing is an explicit user action; downloading works in other browsers.
      if (navigator.canShare && navigator.share && typeof File !== 'undefined') {
        const bytes = Uint8Array.from(atob(url.split(',')[1]), c => c.charCodeAt(0));
        const file = new File([bytes], window.TimelineShare.fileName(finishedState), {type:'image/png'});
        if (navigator.canShare({files:[file]})) {
          const share = make('button', '', '分享');
          share.onclick = async () => {
            try { await navigator.share({files:[file], title:window.TimelineShare.title}); }
            catch (e) { if (e.name !== 'AbortError') notice('分享未完成，可以先保存图片。'); }
          };
          actions.insertBefore(share, again);
        }
      }
    } catch {
      if (!isCurrent()) return;
      status.textContent = '图片生成失败，请重试。';
      const retry = make('button', '', '重试'); retry.onclick = renderFinished; actions.prepend(retry);
    }
  }

  const categoryMap = new Map(cards.map(c => [c.section, c.category]));
  for (const [value, name] of categoryMap) { const option = make('option', '', name); option.value = value; $('#archive-category').append(option); }
  function filterArchive() {
    const q = $('#archive-search').value.trim().toLocaleLowerCase(), cat = $('#archive-category').value;
    archiveCards = cards.filter(c => (!cat || c.section === cat) && (!q || [c.title, c.english, c.id, String(c.year)].some(text => text.toLocaleLowerCase().includes(q))));
    archivePage = 0; renderArchive();
  }
  function renderArchive() {
    const root = $('#archive-grid'), start = archivePage * 24; root.replaceChildren();
    for (const c of archiveCards.slice(start, start + 24)) {
      const tile = make('article', 'archive-tile'), button = make('button', 'archive-face');
      button.setAttribute('aria-label', '查看 ' + c.title + ' 正反面'); button.append(cardImage(c, archiveSide)); button.onclick = () => openCard(c);
      const caption = make('div', 'archive-caption'); caption.append(make('span', '', c.id), make('span', '', c.category)); tile.append(button, caption); root.append(tile);
    }
    if (!archiveCards.length) root.append(make('p', 'muted', '没有匹配的事件，试试别的关键词。'));
    $('#archive-count').textContent = archiveCards.length ? `${archiveCards.length} 张卡片 · 当前 ${start + 1}–${Math.min(start + 24, archiveCards.length)}` : '0 张卡片';
    $('#archive-page').textContent = `${archiveCards.length ? archivePage + 1 : 0} / ${Math.ceil(archiveCards.length / 24)}`;
    $('#archive-prev').disabled = archivePage === 0; $('#archive-next').disabled = start + 24 >= archiveCards.length;
  }
  $('#archive-search').addEventListener('input', filterArchive); $('#archive-category').addEventListener('change', filterArchive);
  for (const side of ['front','back']) $('#archive-' + side).onclick = () => {
    archiveSide = side; for (const s of ['front','back']) $('#archive-' + s).setAttribute('aria-pressed', s === side); renderArchive();
  };
  for (const [button, delta] of [['prev', -1], ['next', 1]]) $('#archive-' + button).onclick = () => {
    archivePage += delta; renderArchive(); $('#archive-search').scrollIntoView({block: 'start', behavior: motion()});
  };
  function openCard(c, frontOnly = false) {
    $('#card-detail').classList.toggle('front-only', frontOnly);
    $('#detail-back').hidden = frontOnly;
    $('.detail-text').hidden = frontOnly;
    $('#detail-title').textContent = `${c.id} · ${c.title}`;
    for (const side of ['front','back']) { $('#detail-' + side).src = asset(c, side); $('#detail-' + side).alt = c.title + (side === 'front' ? '，事件面' : '，答案面'); }
    $('#detail-art-note').textContent = c.art_note;
    $('#detail-sources').replaceChildren();
    for (const source of [{title:'在原作中查看这个事件', url:c.map_source}, ...(c.sources || [])]) {
      if (!/^https?:\/\//.test(source.url)) continue;
      const link = make('a', '', source.title); link.href = source.url; link.target = '_blank'; link.rel = 'noopener'; $('#detail-sources').append(link);
    }
    $('#card-detail details').open = false; $('#card-detail').showModal(); $('#card-detail').scrollTop = 0;
  }
  $('#close-detail').onclick = () => $('#card-detail').close();
  $('#card-detail').addEventListener('click', event => {
    if (event.target !== $('#card-detail')) return;
    const rect = event.target.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close();
  });
  function route(initial = false) {
    let page = location.hash.slice(1); if (!['play','archive','about'].includes(page)) page = 'play';
    for (const el of document.querySelectorAll('.view')) el.hidden = el.id !== 'view-' + page;
    for (const link of document.querySelectorAll('[data-route]')) {
      if (link.dataset.route === page) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    }
    if (page === 'play') renderPlay();
    if (page === 'archive') renderArchive();
    document.title = (page === 'play' ? '' : ({archive:'历史卡', about:'关于'})[page] + ' · ') + '计算帝国';
    if (!initial) { window.scrollTo({top:0, behavior:'instant'}); $('#main').focus({preventScroll:true}); }
  }
  window.addEventListener('hashchange', () => route());
  route(true);
})();
