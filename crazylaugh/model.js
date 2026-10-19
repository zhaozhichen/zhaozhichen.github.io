(function(root){
  'use strict';
  const parameters = [
    ['I','提早知道',5,'ta','告诉','假设 ta 日后仍会知道，告知的价值在于把这段时间提前：ta 可以早点整理情绪、想清需求，决定何时与对象谈、是否继续关系。只计提前获得准备与选择空间的价值，不把同一段时间里避免的金钱或投入损失再算一次。'],
    ['S','情感支持',3,'ta','告诉','告知如果伴随倾听、理解和实际帮助，可能让 ta 感到有人站在自己身边，不必独自处理。例如陪 ta 梳理情况，但不催促分手。这里指这次告知新增的情感支持，不包含原本就会提供的日常陪伴。'],
    ['H','隐私暴露的羞耻',5,'ta','告诉','ta 意识到亲密关系中的隐私已被我知道，可能感到难堪、失去体面，或担心被同情、议论。即使我没有向别人传播，这种暴露感也可能存在。只计隐私被我知晓带来的额外感受，不计出轨事件本身的痛苦，也不计被要求表态的压力。'],
    ['R','揭伤口与表态压力',2,'ta','告诉','主动提起此事，可能让 ta 被迫重温痛苦，或觉得必须向我解释、回应、做决定。例如明明还没想好，却觉得不分手就会被我看不起。这里计算谈话带来的额外压力，与隐私被我知道所产生的暴露感不同。'],
    ['F','迁怒后失去支持',1,'ta','告诉','如果 ta 因迁怒而疏远我，可能同时失去原本能从我这里获得的陪伴、建议或实际帮助。这一项算的是 ta 失去支持的代价；我被责怪、被疏远的代价另计。只计额外损失，不再把本次未能得到的情感支持重复扣一遍。'],
    ['D','晚知道期间的损失',4,'ta','不告诉','在我沉默到 ta 日后知情的这段时间里，ta 可能继续投入金钱、时间或关系承诺，从而多承受一些本可避免的具体损失。例如继续共同出资或推进婚事。这里只计算延迟知情期间新增的具体损失，不包含更早获得准备与选择空间的价值。'],
    ['L','被隐瞒的背叛感',2,'ta','不告诉','如果 ta 后来发现我早已知情却没有说，可能感到被排除、被蒙在鼓里，甚至觉得又受到一次背叛。这一项算 ta 自己的情绪伤害。这种伤害来自对朋友隐瞒的感受，与对象本身的背叛所造成的痛苦不同。'],
    ['T','获得朋友信任',2,'我','告诉','坦诚告知可能让我在 ta 心中更可靠：遇到难开口的事也愿意提醒，而不是只说好听的话。这里计我因告知新增的信任，不是原有友情的全部价值。例如 ta 因此更愿意向我倾诉，或在重要决定前征求我的意见。'],
    ['M','基本迁怒',3,'我','告诉','即使我没有造成坏事，也可能因传递坏消息受到责怪、冷淡或疏远，例如被认为多管闲事、破坏气氛。这是我承担的基本迁怒成本。它来自我作为坏消息传递者受到的负面评价，不包含首次获知时的意外冲击所增加的反感。'],
    ['J','首次获知的额外迁怒',1,'我','告诉','当 ta 首次获知消息且毫无准备时，震惊可能在基本迁怒之外，再增加对我的反感。只计算比基本迁怒多出来的部分。例如突如其来的消息让 ta 一时难以接受，把一部分冲击转向我。首次知道并不必然意味着特别意外。'],
    ['N','损失朋友信任',2,'我','不告诉','如果 ta 后来认为我知情不说是在隐瞒、冷漠或包庇，可能不再把我当作可靠的朋友，减少倾诉或往来。这里计我失去信任的关系成本；ta 自己的背叛感另计。它体现的是 ta 对我的可靠性评价和今后与我相处方式的变化。'],
    ['K','知情不说的心理负担',1,'我','不告诉','即使 ta 从未发现我知情，保守秘密也可能让我内疚、担心说漏嘴，或在与双方相处时反复纠结。例如每次见面都要回避话题。这里只计我自己的心理消耗，不包含被发现后失去信任的损失。']
  ].map(([key,label,value,owner,action,description])=>({key,label,value,owner,action,description}));
  const defaults=Object.fromEntries(parameters.map(p=>[p.key,p.value]));
  const rows=[
    '我知道ta知道','我知道ta不知道','ta知道，我拿不准','ta不知道，我拿不准',
    'ta知道，我误以为ta不知道','ta不知道，我误以为ta知道','ta知道，我正确猜测ta知道','ta不知道，我正确猜测ta不知道'
  ];
  const cols=[
    'ta知道我知道','ta知道我不知道','我知道，ta拿不准','我不知道，ta拿不准',
    '我知道，ta误以为我不知道','我不知道，ta误以为我知道','我知道，ta正确猜测我知道','我不知道，ta正确猜测我不知道'
  ];
  function invalid(table,r,c){
    const reasons=[];
    if ((r%2===0)!==(table==='A')) reasons.push('ta的实际状态与本表矛盾');
    if(c%2===1) reasons.push('要求我实际不知道，不符合题干');
    if(table==='B' && c===0) reasons.push('ta若知道我知道，就已知道该事实');
    return reasons.join('；');
  }
  const cases=[];
  for(const table of ['A','B']){
    let n=0;
    for(let r=0;r<8;r++)for(let c=0;c<8;c++){
      if(invalid(table,r,c))continue;
      const b=table==='B', omitBurden=b&&r===5;
      const ft=[],fs=[],mt=[[1,'T'],[-1,'M']],ms=[[-1,'N']];
      if(b)ft.push([1,'I']);ft.push([1,'S']);
      if(c!==0)ft.push([-1,'H']);ft.push([-1,'R'],[-1,'F']);
      if(b)fs.push([-1,'D']);fs.push([-1,'L']);
      if(b)mt.push([-1,'J']);if(!omitBurden)ms.push([-1,'K']);
      const group=!b?(c===0?0:1):(omitBurden?3:2);
      cases.push({id:table+(++n),table,r,c,group,ft,fs,mt,ms});
    }
  }
  function evaluate(terms,p){return terms.reduce((s,[sign,key])=>s+sign*p[key],0)}
  function score(item,p){const x=Object.fromEntries(['ft','fs','mt','ms'].map(k=>[k,evaluate(item[k],p)]));return {...x,tell:x.ft+x.mt,silent:x.fs+x.ms,delta:x.ft+x.mt-x.fs-x.ms}}
  function format(terms){return terms.map(([s,k],i)=>(s<0?'−':i?'＋':'')+k).join('')||'0'}
  function select({table='all',row='all',col='all',id=null}={}){return cases.filter(c=>(!id||c.id===id)&&(table==='all'||c.table===table)&&(row==='all'||c.r===Number(row))&&(col==='all'||c.c===Number(col)))}
  function aggregate(items,p){if(!items.length)return null;const keys=['ft','fs','mt','ms','tell','silent','delta'];const x=Object.fromEntries(keys.map(k=>[k,0]));const wins={tell:0,silent:0,tie:0};for(const item of items){const s=score(item,p);for(const k of keys)x[k]+=s[k]/items.length;wins[Math.abs(s.delta)<1e-8?'tie':s.delta>0?'tell':'silent']++}return {...x,wins,count:items.length}}
  const api={parameters,defaults,rows,cols,cases,invalid,score,format,select,aggregate};
  root.DebateModel=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
