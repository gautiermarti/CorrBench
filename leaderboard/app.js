"use strict";
const byId = id => document.getElementById(id);
let data, selected, sortKey, sortAscending;
for (let i=0;i<16;i++) for (let j=0;j<16;j++) {
  const cell=document.createElement("div");
  const same=Math.floor(i/4)===Math.floor(j/4);
  const correlation=i===j?1:(same ? .64 : .14)+.12*Math.exp(-Math.abs(i-j)/4);
  cell.style.background="hsl(171 36% "+(94-correlation*64)+"%)";
  byId("matrix").appendChild(cell);
}
function format(value){return value===undefined||value===null?"—":Number.isInteger(value)?String(value):value.toFixed(4);}
function showProtocol(id){
  selected=data.protocols.find(p=>p.id===id)||data.protocols[0];
  byId("protocol").value=selected.id;
  sortKey=selected.metric; sortAscending=selected.direction==="lower";
  byId("domain").textContent=selected.domain;
  byId("protocol-title").textContent=selected.title;
  byId("status").textContent=selected.status;
  byId("description").textContent=selected.description;
  byId("finding").textContent=selected.finding;
  byId("limitations").textContent=selected.limitations;
  byId("table-caption").textContent=selected.title+" — "+selected.metric_label;
  byId("protocol-links").replaceChildren();
  for(const [path,label] of [[selected.protocol_file,"Read protocol ↗"],[selected.evidence_file,"Inspect evidence ↗"]]){
    if(!path)continue;
    const link=document.createElement("a");link.href=path;link.textContent=label;
    link.target="_blank";link.rel="noopener";byId("protocol-links").appendChild(link);
  }
  byId("dataset-hash").textContent="DATASET SHA-256 / "+(selected.dataset_hash||"Awaiting complete audit");
  const notes=byId("record-notes"); notes.replaceChildren();
  for(const row of data.records.filter(r=>r.protocol===selected.id)){
    const li=document.createElement("li");
    li.textContent=row.method+": "+row.note;notes.appendChild(li);
  }
  const url=new URL(location.href);url.searchParams.set("protocol",selected.id);
  history.replaceState(null,"",url);
  renderTable();
}
function renderTable(){
  const head=byId("table-head"),body=byId("table-body");head.replaceChildren();body.replaceChildren();
  const header=document.createElement("tr"),name=document.createElement("th");name.scope="col";name.textContent="Method";header.appendChild(name);
  for(const [key,label] of selected.columns){
    const th=document.createElement("th");th.scope="col";
    const button=document.createElement("button");button.className="sort-button";
    button.textContent=label+(sortKey===key?(sortAscending?" ▴":" ▾"):"");
    button.setAttribute("aria-label","Sort by "+label);
    button.addEventListener("click",()=>{sortAscending=sortKey===key?!sortAscending:!label.includes("↑");sortKey=key;renderTable();});
    if(sortKey===key)th.setAttribute("aria-sort",sortAscending?"ascending":"descending");
    th.appendChild(button);header.appendChild(th);
  }
  head.appendChild(header);
  const rows=data.records.filter(r=>r.protocol===selected.id);
  rows.sort((a,b)=>{
    const x=a.metrics[sortKey],y=b.metrics[sortKey];
    if(x===undefined)return y===undefined?0:1;if(y===undefined)return -1;
    return (x-y)*(sortAscending?1:-1);
  });
  if(!rows.length){
    const tr=document.createElement("tr"),td=document.createElement("td");
    td.colSpan=selected.columns.length+1;td.className="empty";
    td.textContent="Results will appear after all required seeds and provenance checks finish. Partial runs are not ranked.";
    tr.appendChild(td);body.appendChild(tr);return;
  }
  for(const row of rows){
    const tr=document.createElement("tr"),td=document.createElement("td");
    td.textContent=row.method;tr.appendChild(td);
    for(const [key] of selected.columns){const cell=document.createElement("td");cell.textContent=format(row.metrics[key]);tr.appendChild(cell);}
    body.appendChild(tr);
  }
}
byId("protocol").addEventListener("change",event=>showProtocol(event.target.value));
byId("download").addEventListener("click",()=>{
  const payload={protocol:selected,records:data.records.filter(r=>r.protocol===selected.id),review_scope:data.review_scope};
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)+"\n"],{type:"application/json"}));
  const a=document.createElement("a");a.href=url;a.download=selected.id+".json";a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.querySelectorAll(".protocol-link").forEach(button=>button.addEventListener("click",()=>{
  if(!data)return;showProtocol(button.dataset.protocol);byId("results").scrollIntoView({behavior:"smooth"});
}));
fetch("results.json").then(response=>{if(!response.ok)throw Error("Missing result records");return response.json();}).then(payload=>{
  data=payload;
  for(const p of data.protocols){const option=document.createElement("option");option.value=p.id;option.textContent=p.title;byId("protocol").appendChild(option);}
  showProtocol(new URL(location.href).searchParams.get("protocol")||"abide_utility_20261005");
}).catch(error=>{byId("finding").textContent="Could not load the local results: "+error.message;});
