'use strict';
const M=globalThis.DebateModel;
let values={...M.defaults};
let filter={table:'all',row:'all',col:'all',id:null};
const $=s=>document.querySelector(s);
const number=x=>{if(Math.abs(x)<1e-8)x=0;return Number.isInteger(x)?String(x):x.toFixed(2)};
const signed=x=>(x>1e-8?'+':'')+number(x);
const preference=x=>Math.abs(x)<1e-8?'tie':x>0?'tell':'silent';
const preferenceLabel=x=>Math.abs(x)<1e-8?'两者持平':x>0?'告诉更优':'不告诉更优';
const groups=[['ta的收益与成本',p=>p.owner==='ta'&&p.action==='告诉'],['ta在沉默中的损失',p=>p.owner==='ta'&&p.action==='不告诉'],['我的收益与成本',p=>p.owner==='我']];
$('#sliders').innerHTML=groups.map(([title,predicate])=>`<h3 class="parameter-group">${title}</h3>${M.parameters.filter(predicate).map(p=>`<div class="parameter"><div class="parameter-top"><span class="symbol">${p.key}</span><label for="param-${p.key}" title="${p.description}">${p.label}</label><output id="output-${p.key}" for="param-${p.key}">${p.value}</output></div><input id="param-${p.key}" type="range" min="0" max="10" step="0.5" value="${p.value}" aria-describedby="desc-${p.key}"><p id="desc-${p.key}" class="description">${p.action} · ${p.description}</p></div>`).join('')}`).join('');
document.querySelector('.parameters').id='parameters';
document.querySelector('.selection').insertAdjacentHTML('beforebegin','<a class="mobile-parameters" href="#parameters">调节14项收益与成本 ↓</a>');
const visibleCols=[0,2,4,6];
for(const [selector,labels,indices] of [['#row-filter',M.rows,M.rows.map((_,i)=>i)],['#col-filter',M.cols,visibleCols]]){
  $(selector).insertAdjacentHTML('beforeend',indices.map(i=>`<option value="${i}">${labels[i]}</option>`).join(''));
}
function matrix(){return `<section class="matrix-section" id="matrix"><div class="matrix-title"><h3>我已知情 · 全部28种情况</h3></div><div class="scroll" tabindex="0" aria-label="8行4列知情状态矩阵"><table class="matrix"><thead><tr><th class="corner" scope="col">列：ta对我的判断 →<br>行：我对ta的判断 ↓</th>${visibleCols.map(c=>`<th scope="col"><button class="axis-button" data-col="${c}" aria-pressed="false">${M.cols[c]}</button></th>`).join('')}</tr></thead><tbody>${M.rows.map((s,r)=>{const table=r%2===0?'A':'B';return `<tr><th scope="row"><button class="axis-button" data-row="${r}" aria-pressed="false">${s}</button></th>${visibleCols.map(c=>{const reason=M.invalid(table,r,c);if(reason)return `<td><div class="invalid" title="${reason}"><div class="cross" aria-hidden="true">×</div><div class="invalid-label">不成立</div><span>${reason}</span></div></td>`;const item=M.cases.find(x=>x.r===r&&x.c===c);return `<td><button class="cell" data-case="${item.id}" aria-pressed="false"></button></td>`}).join('')}</tr>`}).join('')}</tbody></table></div></section>`}
$('#matrices').innerHTML=matrix();
function formula(item,key,score){return `<span class="formula-line"><span>${key[0]==='f'?'ta':'我'}：${M.format(item[key])}</span><strong>${number(score[key])}</strong></span>`}
function render(){
  const items=M.select(filter),active=new Set(items.map(x=>x.id)),sum=M.aggregate(items,values);
  $('#row-filter').value=filter.row;$('#col-filter').value=filter.col;
  document.querySelectorAll('[data-scope]').forEach(b=>b.setAttribute('aria-pressed',String(!filter.id&&b.dataset.scope===filter.table)));
  document.querySelectorAll('[data-row]').forEach(b=>b.setAttribute('aria-pressed',String(!filter.id&&b.dataset.row===filter.row)));
  document.querySelectorAll('[data-col]').forEach(b=>b.setAttribute('aria-pressed',String(!filter.id&&b.dataset.col===filter.col)));
  for(const item of M.cases){const s=M.score(item,values),b=document.querySelector(`[data-case="${item.id}"]`);b.className=`cell ${preference(s.delta)}${active.has(item.id)?'':' excluded'}`;b.setAttribute('aria-pressed',String(filter.id===item.id));b.setAttribute('aria-label',`${item.id}，${M.rows[item.r]}；${M.cols[item.c]}。告诉总分${number(s.tell)}，不告诉总分${number(s.silent)}。点击单独计算`);b.innerHTML=`<span class="cell-heading"><span>${item.id}</span><em>${preferenceLabel(s.delta)}</em></span><span class="formula-group"><b class="tell-text">告诉</b>${formula(item,'ft',s)}${formula(item,'mt',s)}</span><span class="formula-group"><b class="silent-text">不告诉</b>${formula(item,'fs',s)}${formula(item,'ms',s)}</span><span class="cell-totals"><span>告诉 ${number(s.tell)}</span><span>不告诉 ${number(s.silent)}</span></span>`}
  if(!sum){$('#summary').innerHTML='<div class="empty"><h2>这个筛选下没有有效情形</h2><p>当前条件互相矛盾。请放宽行列筛选，或重置为全部28种；无效情形不会计入平均。</p><button data-scope="all">重置为全部28种</button></div>';$('#detail').innerHTML='';return}
  const description=filter.id?`只看 ${filter.id}`:[filter.table==='all'?'全部一阶状态':filter.table==='A'?'ta已知':'ta未知',filter.row==='all'?'我对ta的判断不限':M.rows[Number(filter.row)],filter.col==='all'?'ta对我的判断不限':M.cols[Number(filter.col)]].join(' / ');
  const pref=preference(sum.delta),label=preferenceLabel(sum.delta);
  const differences=[sum.ft-sum.fs,sum.mt-sum.ms,sum.delta];
  $('#summary').innerHTML=`<div class="summary-top"><div><h2>${items.length===1?'这一种情况的分数':'当前范围的平均分'}</h2><p class="scope-caption">${description}</p></div><div class="verdict ${pref}-text">${label}<small>${pref==='tie'?'净差 0':`领先 ${number(Math.abs(sum.delta))} 分`}</small></div></div><table class="score-table"><thead><tr><th>谁的分数</th><th class="tell-text">告诉</th><th class="silent-text">不告诉</th><th>告诉 − 不告诉</th></tr></thead><tbody><tr><td>对ta</td><td>${number(sum.ft)}</td><td>${number(sum.fs)}</td><td>${signed(differences[0])}</td></tr><tr><td>对我</td><td>${number(sum.mt)}</td><td>${number(sum.ms)}</td><td>${signed(differences[1])}</td></tr><tr class="total"><td>双方合计</td><td>${number(sum.tell)}</td><td>${number(sum.silent)}</td><td class="${pref}-text">${signed(sum.delta)}</td></tr></tbody></table><div class="distribution"><span><i class="dot tell"></i> 告诉更优 ${sum.wins.tell}种</span><span><i class="dot silent"></i> 不告诉更优 ${sum.wins.silent}种</span><span><i class="dot tie"></i> 持平 ${sum.wins.tie}种</span><span class="prob">${sum.count}种有效情形 · 每种概率 1/${sum.count}（${number(100/sum.count)}%）</span></div>`;
  if(items.length===1){const item=items[0],s=M.score(item,values);const numeric=terms=>terms.map(([sign,k],i)=>(sign<0?'−':i?'＋':'')+values[k]).join('');$('#detail').innerHTML=`<section class="detail"><h3>${item.id} · 逐项展开</h3><p>${M.rows[item.r]}；${M.cols[item.c]}。</p><div class="detail-formulas">${[['告诉','ft','mt'],['不告诉','fs','ms']].map(([title,...keys])=>`<div><strong>${title}</strong>${keys.map(k=>`<p>${k[0]==='f'?'对ta':'对我'}：${M.format(item[k])}<br>${numeric(item[k])} = <strong>${number(s[k])}</strong></p>`).join('')}</div>`).join('')}</div></section>`}else $('#detail').innerHTML='';
}
$('#sliders').addEventListener('input',e=>{if(!e.target.matches('input[type=range]'))return;const key=e.target.id.replace('param-','');values[key]=Number(e.target.value);$(`#output-${key}`).value=number(values[key]);render()});
function setValues(next){values={...next};for(const p of M.parameters){$(`#param-${p.key}`).value=values[p.key];$(`#output-${p.key}`).value=number(values[p.key])}render()}
$('#reset').addEventListener('click',()=>setValues(M.defaults));
$('#zero').addEventListener('click',()=>setValues(Object.fromEntries(M.parameters.map(p=>[p.key,0]))));
$('#clear').addEventListener('click',()=>{filter={table:'all',row:'all',col:'all',id:null};render()});
$('#row-filter').addEventListener('change',e=>{filter={...filter,row:e.target.value,id:null};render()});
$('#col-filter').addEventListener('change',e=>{filter={...filter,col:e.target.value,id:null};render()});
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.scope){filter={table:b.dataset.scope,row:'all',col:'all',id:null};render()}else if(b.dataset.case){filter={table:'all',row:'all',col:'all',id:b.dataset.case};render();$('#summary').scrollIntoView({block:'start',behavior:'instant'})}else if(b.dataset.row!==undefined){filter={table:'all',row:b.dataset.row,col:'all',id:null};render();$('#summary').scrollIntoView({block:'start',behavior:'instant'})}else if(b.dataset.col!==undefined){filter={table:filter.table,row:'all',col:b.dataset.col,id:null};render();$('#summary').scrollIntoView({block:'start',behavior:'instant'})}});
render();
