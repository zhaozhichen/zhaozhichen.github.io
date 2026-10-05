(function () {
  'use strict';
  const TITLE = '计算帝国——时间线';
  const sans = '"PingFang SC", "Microsoft YaHei", sans-serif';
  const titleFont = '"Timeline Title Heavy"';
  const nameFont = '"Timeline Name Heavy", "Songti SC", serif';
  function timestamp(time) {
    const d = new Date(time), pad = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }
  async function make(state) {
    // Canvas snapshots do not update when a webfont arrives: load it first.
    const [titleFaces] = await Promise.all([
      document.fonts.load(`900 43px ${titleFont}`, TITLE),
      document.fonts.load('900 48px "Timeline Name Heavy"', state.name)
    ]);
    if (!titleFaces.length) throw new Error('The title font is unavailable.');
    const score = window.TimelineGame.stats(state), canvas = document.createElement('canvas');
    canvas.width = 1080; canvas.height = 1350;
    const ctx = canvas.getContext('2d');
    const text = (value, x, y, size, color = '#fff', weight = '400', maxWidth = 920, family = sans) => {
      ctx.fillStyle = color; ctx.font = `${weight} ${size}px ${family}`;
      while (ctx.measureText(value).width > maxWidth && size > 12) ctx.font = `${weight} ${--size}px ${family}`;
      ctx.fillText(value, x, y);
    };
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 1080, 1350);
    ctx.strokeStyle = '#666'; ctx.lineWidth = 1; ctx.strokeRect(32.5, 32.5, 1015, 1285);
    text(TITLE, 80, 120, 43, '#fff', '900', 920, titleFont);
    text(timestamp(state.endedAt), 80, 167, 23, '#aaa', '400', 920, 'Arial, sans-serif');
    ctx.strokeStyle = '#555'; ctx.beginPath(); ctx.moveTo(80, 210); ctx.lineTo(1000, 210); ctx.stroke();
    text(state.name, 80, 310, 48, '#fff', '900', 920, nameFont);
    text((score.accuracy * 100).toFixed(1) + '%', 66, 538, 206, '#fff', '600', 930, 'Arial, sans-serif');
    text('正确率', 82, 592, 28, '#bbb');
    text(`答对 ${score.correct} / ${score.answered} 张`, 80, 696, 38);
    // A small timeline visual also represents the last ten answers.
    const recent = state.attempts.slice(-10), start = 104, end = 976, y = 882;
    ctx.strokeStyle = '#888'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(80, y); ctx.lineTo(1000, y); ctx.stroke();
    recent.forEach((answer, i) => {
      const x = start + i * (end - start) / Math.max(1, recent.length - 1);
      ctx.fillStyle = answer.correct ? '#fff' : '#000'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    });
    const matrix = window.TIMELINE_QR.matrix, unit = Math.floor(276 / matrix.length), size = unit * matrix.length;
    const qrX = 1000 - size, qrY = 960;
    ctx.fillStyle = '#fff'; ctx.fillRect(qrX, qrY, size, size);
    ctx.fillStyle = '#000';
    matrix.forEach((row, y) => row.forEach((dark, x) => { if (dark) ctx.fillRect(qrX + x * unit, qrY + y * unit, unit, unit); }));
    text('你来试试', 80, 1075, 49, '#fff', '500', 600);
    text('zhaozhichen.github.io/timeline-game/', 80, 1195, 22, '#bbb', '400', 615, 'Arial, sans-serif');
    return canvas;
  }
  function fileName(state) {
    const name = state.name.replace(/[\\/:*?"<>|\u0000-\u001f]/g, '_');
    return `计算帝国-时间线-${name}-${state.endedAt}.png`;
  }
  window.TimelineShare = {make, timestamp, fileName, title: TITLE};
})();
