(function(root){
  'use strict';
  const parameters = [
    ['I','提早知道',4,'ta','告诉','假设 ta 日后仍会知道，告知的价值在于把这段时间提前：ta 可以早点整理情绪、想清需求，决定何时与对象谈、是否继续关系。只计提前获得准备与选择空间的价值，不把同一段时间里避免的金钱或投入损失再算一次。'],
    ['S','情感支持',4,'ta','告诉','告知如果伴随倾听、理解和实际帮助，可能让 ta 感到有人站在自己身边，不必独自处理。例如陪 ta 梳理情况，但不催促分手。只计这次告知新增的支持；若只是丢下一句爆料，或本来就会提供这些支持，可以调低。'],
    ['H','隐私暴露的羞耻',6,'ta','告诉','ta 意识到亲密关系中的隐私已被我知道，可能感到难堪、失去体面，或担心被同情、议论。即使我没有向别人传播，这种暴露感也可能存在。只计隐私被我知晓带来的额外感受，不计出轨事件本身的痛苦，也不计被要求表态的压力。'],
    ['R','揭伤口与表态压力',3,'ta','告诉','主动提起此事，可能让 ta 被迫重温痛苦，或觉得必须向我解释、回应、做决定。例如明明还没想好，却觉得不分手就会被我看不起。这里计算谈话带来的额外压力；若能尊重 ta 的节奏、不追问也不催促，可以调低。'],
    ['F','迁怒后失去支持',1,'ta','告诉','如果 ta 因迁怒而疏远我，可能同时失去原本能从我这里获得的陪伴、建议或实际帮助。这一项算的是 ta 失去支持的代价；我被责怪、被疏远的代价另计。只计额外损失，不再把本次未能得到的情感支持重复扣一遍。'],
    ['D','晚知道期间的损失',3,'ta','不告诉','在我沉默到 ta 日后知情的这段时间里，ta 可能继续投入金钱、时间或关系承诺，从而多承受一些本可避免的具体损失。例如继续共同出资或推进婚事。只算这段延迟新增的损失；如果很快会知道，或期间不会作出重要投入，可以调低。'],
    ['L','被隐瞒的背叛感',1,'ta','不告诉','如果 ta 后来发现我早已知情却没有说，可能感到被排除、被蒙在鼓里，甚至觉得又受到一次背叛。这一项算 ta 自己的情绪伤害。数值应同时考虑 ta 发现我隐瞒的可能性和发现后有多在意，不是假定沉默一定会被识破。'],
    ['T','获得朋友信任',2,'我','告诉','坦诚告知可能让我在 ta 心中更可靠：遇到难开口的事也愿意提醒，而不是只说好听的话。这里计我因告知新增的信任，不是原有友情的全部价值。如果 ta 并不期待朋友通报，或认为这种提醒越界，这项收益可以很低。'],
    ['M','基本迁怒',4,'我','告诉','即使我没有造成坏事，也可能因传递坏消息受到责怪、冷淡或疏远，例如被认为多管闲事、破坏气氛。这是我承担的基本迁怒成本。结合反应出现的可能性与影响大小设值，不意味着朋友必然会迁怒，也不把对方最严重的可能反应当成常态。'],
    ['J','首次获知的额外迁怒',2,'我','告诉','当 ta 首次获知消息且毫无准备时，震惊可能在基本迁怒之外，再增加对我的反感。只计算比基本迁怒多出来的部分。如果 ta 早有怀疑、已有心理准备，或我能妥善铺垫，可以调低甚至设为 0；首次知道并不必然意味着特别意外。'],
    ['N','损失朋友信任',1,'我','不告诉','如果 ta 后来认为我知情不说是在隐瞒、冷漠或包庇，可能不再把我当作可靠的朋友，减少倾诉或往来。这里计我失去信任的关系成本；ta 自己的背叛感另计。应考虑沉默被发现的可能性，以及 ta 原本是否期待我主动告知。'],
    ['K','知情不说的心理负担',1,'我','不告诉','即使 ta 从未发现我知情，保守秘密也可能让我内疚、担心说漏嘴，或在与双方相处时反复纠结。例如每次见面都要回避话题。这里只计我自己的心理消耗，不包含被发现后失去信任的损失；若能接受自己的边界，负担可能很低。']
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
