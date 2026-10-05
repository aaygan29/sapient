"""Layer 4 — visualization.

Compiles a Timeline + Stats into a self-contained, interactive HTML report featuring a
LIVE DIGITAL BRAIN: the 7 Yeo-7 cortical networks light up second-by-second as the media
plays, with synced metric bars (persuasion / buy-sell / arousal / engagement / valence),
a scrubbable persuasion timeline, the "moments that matter", and a per-person switcher
(average brain vs enrolled subjects).

    from neurosignal.viz import render_report, save_report
    save_report("report.html", {"Average brain": {"timeline": tl.to_dict(), "stats": st.to_dict()}})
"""
from __future__ import annotations

import json

_TEMPLATE = r"""<div id="ns-root">
<style>
#ns-root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--line:#30363d;
  background:var(--bg);color:var(--ink);border-radius:14px;padding:18px 20px;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:980px;margin:auto}
#ns-root h1{font-size:19px;margin:0 0 2px}#ns-root .sub{color:var(--muted);font-size:13px;margin-bottom:10px}
#ns-root .headline{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:9px 12px;font-size:13px;margin-bottom:12px}
#ns-root .subjects{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
#ns-root .subjects button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:20px;padding:5px 13px;font-size:12px;cursor:pointer}
#ns-root .subjects button.active{background:#1f6feb;border-color:#1f6feb;color:#fff}
#ns-root .grid{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}
@media(max-width:760px){#ns-root .grid{grid-template-columns:1fr}}
#ns-root .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
#ns-root .metric{margin:9px 0}#ns-root .metric .lab{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px}
#ns-root .bar{height:9px;background:#0d1117;border-radius:6px;overflow:hidden}
#ns-root .bar>span{display:block;height:100%;width:0;transition:width .12s linear}
#ns-root .controls{display:flex;align-items:center;gap:10px;margin-top:10px}
#ns-root .controls button{background:#1f6feb;border:none;color:#fff;border-radius:8px;padding:6px 14px;cursor:pointer;font-size:13px}
#ns-root input[type=range]{flex:1;accent-color:#1f6feb}
#ns-root .time{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px;min-width:62px;text-align:right}
#ns-root .rec{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}
#ns-root .moments{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.6}
#ns-root .moments b{color:var(--ink)}
#ns-root text{font-size:9px;fill:var(--ink);font-family:inherit}
</style>

<h1 id="ns-title"></h1><div class="sub" id="ns-sub"></div>
<div class="headline" id="ns-headline"></div>
<div class="subjects" id="ns-subjects"></div>

<div class="grid">
  <div class="card">
    <svg id="ns-brain" viewBox="0 0 440 320" width="100%"></svg>
    <div style="margin-top:8px">
      <svg id="ns-spark" viewBox="0 0 440 80" width="100%"></svg>
    </div>
    <div class="controls">
      <button id="ns-play">▶ Play</button>
      <input id="ns-scrub" type="range" min="0" value="0"/>
      <span class="time" id="ns-time">0.0s</span>
    </div>
  </div>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
      <span style="font-size:13px;font-weight:600">Read-out at this moment</span>
      <span class="rec" id="ns-rec">Hold</span>
    </div>
    <div id="ns-metrics"></div>
    <div class="moments" id="ns-momlist"></div>
  </div>
</div>
</div>
<script>
(function(){
const DATA = __DATA__;
const NETS = {
  Frontoparietal:{x:108,y:120,rx:44,ry:36,t:"Frontoparietal",s:"analytic control"},
  Limbic:{x:120,y:206,rx:42,ry:30,t:"Limbic",s:"reward / emotion"},
  VentralAttention:{x:206,y:210,rx:40,ry:28,t:"Ventral Attn",s:"salience / urgency"},
  Somatomotor:{x:208,y:92,rx:48,ry:26,t:"Somatomotor",s:"action"},
  DorsalAttention:{x:284,y:120,rx:44,ry:32,t:"Dorsal Attn",s:"focus"},
  Default:{x:236,y:166,rx:42,ry:30,t:"Default",s:"memory / self"},
  Visual:{x:356,y:166,rx:46,ry:42,t:"Visual",s:"seeing"}
};
const METRICS=[
  {k:"manipulation",l:"Persuasion Pressure",c:"#ff5c5c"},
  {k:"buy_sell",l:"Buy / Sell Signal",c:"#4cd964"},
  {k:"arousal",l:"Arousal / Intensity",c:"#ffb74d"},
  {k:"engagement",l:"Engagement",c:"#5ac8fa"},
  {k:"valence",l:"Affective Valence",c:"#b388ff",signed:true}
];
const $=id=>document.getElementById(id);
$("ns-title").textContent=DATA.title; $("ns-sub").textContent=DATA.subtitle||"";

let subject=DATA.subjects[0], idx=0, playing=false, timer=null;

// build brain svg
const brain=$("ns-brain");
brain.innerHTML='<path d="M58,156 C58,84 132,46 224,46 C326,46 404,82 404,162 C404,238 332,278 230,278 C138,278 58,236 58,156 Z" '
  +'fill="#0d1117" stroke="#30363d" stroke-width="2"/>';
for(const [k,r] of Object.entries(NETS)){
  brain.innerHTML+=`<ellipse id="rg-${k}" cx="${r.x}" cy="${r.y}" rx="${r.rx}" ry="${r.ry}" fill="#ff7043" fill-opacity="0.05" stroke="#30363d" stroke-width="1"/>`
   +`<text x="${r.x}" y="${r.y-2}" text-anchor="middle" font-weight="700">${r.t}</text>`
   +`<text x="${r.x}" y="${r.y+9}" text-anchor="middle" fill="#8b949e">${r.s}</text>`;
}
// metric rows
$("ns-metrics").innerHTML=METRICS.map(m=>
  `<div class="metric"><div class="lab"><span>${m.l}</span><span id="mv-${m.k}">–</span></div>`
  +`<div class="bar"><span id="mb-${m.k}" style="background:${m.c}"></span></div></div>`).join("");

function heat(v){ // 0..100 -> warm glow opacity + hue
  const o=0.06+0.0094*v; return o; }
function frames(){return DATA.reports[subject].timeline.frames;}
function stats(){return DATA.reports[subject].stats;}

function drawSpark(){
  const fr=frames(), W=440,H=80, n=fr.length;
  const xs=i=>10+(W-20)*(n<2?0.5:i/(n-1));
  const ys=v=>H-8-(H-16)*(v/100);
  let pts=fr.map((f,i)=>`${xs(i)},${ys(f.metrics.manipulation||0)}`).join(" ");
  $("ns-spark").innerHTML=
    `<text x="10" y="12" fill="#8b949e">Persuasion pressure over time</text>`
    +`<polyline points="${pts}" fill="none" stroke="#ff5c5c" stroke-width="2"/>`
    +`<line id="ns-head" x1="${xs(idx)}" y1="14" x2="${xs(idx)}" y2="${H-6}" stroke="#e6edf3" stroke-width="1" stroke-dasharray="3 3"/>`;
}
function moveHead(){const fr=frames(),n=fr.length,W=440;const x=10+(W-20)*(n<2?0.5:idx/(n-1));
  const h=$("ns-head"); if(h){h.setAttribute("x1",x);h.setAttribute("x2",x);} }

function render(){
  const fr=frames(), f=fr[idx];
  for(const k of Object.keys(NETS)){
    const el=$("rg-"+k); const v=(f.networks[k]!==undefined)?f.networks[k]:0;
    el.setAttribute("fill-opacity", heat(v).toFixed(3));
    el.setAttribute("stroke", v>60?"#ff7043":"#30363d");
  }
  for(const m of METRICS){
    let v=f.metrics[m.k]; if(v===undefined)v=0;
    const w=m.signed?(v+100)/2:v;
    $("mb-"+m.k).style.width=Math.max(0,Math.min(100,w))+"%";
    $("mv-"+m.k).textContent=(m.signed?(v>0?"+":""):"")+v.toFixed(0);
  }
  const rec=f.recommendation||"Hold", rc=$("ns-rec"); rc.textContent=rec;
  rc.style.background=rec==="Buy"?"#11331f":rec==="Sell"?"#3a1414":"#22272e";
  rc.style.color=rec==="Buy"?"#4cd964":rec==="Sell"?"#ff5c5c":"#8b949e";
  $("ns-time").textContent=f.t.toFixed(1)+"s";
  $("ns-scrub").value=idx; moveHead();
}
function setSubject(s){
  subject=s; idx=0;
  for(const b of document.querySelectorAll("#ns-subjects button"))
    b.classList.toggle("active", b.dataset.s===s);
  const st=stats();
  $("ns-headline").innerHTML="<b>"+(DATA.reports[subject].timeline.personalized?"Personalized brain — ":"Average brain — ")+"</b>"+st.headline;
  $("ns-scrub").max=frames().length-1;
  $("ns-momlist").innerHTML="<b>Moments that matter:</b><br>"+
    st.peak_moments.slice(0,4).map(p=>`• ${p.label}: <b>${p.score.toFixed(0)}</b> at ${p.t.toFixed(0)}s`).join("<br>");
  drawSpark(); render();
}
// subject buttons
$("ns-subjects").innerHTML=DATA.subjects.map((s,i)=>
  `<button data-s="${s}" class="${i===0?'active':''}">${s}</button>`).join("");
for(const b of document.querySelectorAll("#ns-subjects button"))
  b.onclick=()=>{stop();setSubject(b.dataset.s);};

$("ns-scrub").oninput=e=>{idx=+e.target.value;render();};
function step(){const n=frames().length; idx=(idx+1)%n; render(); if(idx===n-1)stop();}
function play(){playing=true;$("ns-play").textContent="❚❚ Pause";
  const fps=DATA.reports[subject].timeline.fps||1; timer=setInterval(step,1000/Math.max(fps,0.5)/1.5);}
function stop(){playing=false;$("ns-play").textContent="▶ Play";if(timer)clearInterval(timer);}
$("ns-play").onclick=()=>{playing?stop():(idx>=frames().length-1&&(idx=0),play());};

setSubject(DATA.subjects[0]);
})();
</script>"""


def render_report(reports: dict, title: str = "Neuro Read-out — Live Digital Brain",
                  subtitle: str = "") -> str:
    """reports: {label: {"timeline": timeline.to_dict(), "stats": stats.to_dict()}}"""
    payload = {"title": title, "subtitle": subtitle,
               "subjects": list(reports.keys()), "reports": reports}
    return _TEMPLATE.replace("__DATA__", json.dumps(payload))


def save_report(path: str, reports: dict, **kw) -> str:
    html = render_report(reports, **kw)
    full = "<!doctype html><html><head><meta charset='utf-8'><title>Neuro Read-out</title>" \
           "<body style='margin:0;background:#0d1117'>" + html + "</body></html>"
    with open(path, "w") as fh:
        fh.write(full)
    return path
