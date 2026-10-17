(function(root){
  'use strict';
  const parameters = [
    ['I','提早知道',4.5,'ta','告诉','提前获得准备与自主选择的时间；不计避免的具体损失。'],
    ['S','情感支持',2,'ta','告诉','这次告知实际提供的理解、陪伴与帮助。'],
    ['H','隐私暴露的羞耻',4,'ta','告诉','ta明确发现我知情后的额外羞耻感。'],
    ['R','揭伤口与表态压力',3,'ta','告诉','主动提起带来的被审视、被要求解释的压力。'],
    ['F','迁怒后失去支持',1,'ta','告诉','ta因迁怒我而失去帮助的额外成本。'],
    ['D','晚知道期间的损失',3,'ta','不告诉','从现在到日后知情之间，多承受的投入或受骗损失；不重复计入准备与选择的价值。'],
    ['L','被隐瞒的背叛感',1,'ta','不告诉','ta察觉我知情未说而受伤的预期成本。'],
    ['T','获得朋友信任',2,'我','告诉','告知使我获得的新增信任。'],
    ['M','基本迁怒',4,'我','告诉','报信者受到责怪、疏远的预期成本。'],
    ['J','首次获知的额外迁怒',2,'我','告诉','坏消息带来的意外冲击，使报信者承受更多迁怒。'],
    ['N','损失朋友信任',1,'我','不告诉','沉默被视为隐瞒、冷漠或包庇的预期成本。'],
    ['K','知情不说的心理负担',1,'我','不告诉','保守秘密带来的担忧、内疚与内耗。']
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
