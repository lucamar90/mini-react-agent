"""A tiny local web UI that streams the agent's reasoning live (SSE).

Run it with ``mini-agent serve`` (needs the ``web`` extra: ``pip install
'mini-react-agent[web]'``). The page lets you type a task, pick a backend
(offline demo or live Claude/OpenAI), and watch each Thought / Action /
Observation appear as the agent produces it.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from .agent import Agent
from .demo import load_demo
from .llms import get_llm
from .tools import default_tools


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so a local key 'just works' (no extra dependency)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _stream_agent(task: str, backend: str, model: str | None = None):
    """Generator of Server-Sent Events as the agent runs (in a worker thread)."""
    events: queue.Queue = queue.Queue()
    # The offline demo returns instantly; pace it so the UI still animates.
    delay = 0.6 if backend == "demo" else 0.0

    def on_step(index, step):
        events.put(
            {
                "type": "step",
                "index": index,
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
            }
        )
        if delay:
            time.sleep(delay)

    def worker():
        try:
            if backend == "demo":
                use_task, outputs = load_demo()
                llm = get_llm("replay", outputs=outputs)
            else:
                use_task = task
                llm = get_llm(backend, model=model)
            events.put({"type": "start", "task": use_task, "backend": llm.name, "model": llm.model})
            result = Agent(llm, default_tools(), max_steps=8, on_step=on_step).run(use_task)
            events.put(
                {
                    "type": "done",
                    "final_answer": result.final_answer,
                    "stopped_reason": result.stopped_reason,
                    "succeeded": result.succeeded,
                }
            )
        except Exception as exc:  # surface backend/key errors to the UI
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = events.get()
        if item is None:
            break
        yield f"data: {json.dumps(item)}\n\n"


def create_app():
    from flask import Flask, Response, request

    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/health")
    def health():
        return {"ok": True, "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "openai_key": bool(os.environ.get("OPENAI_API_KEY"))}

    @app.get("/run")
    def run():
        task = request.args.get("task", "").strip()
        backend = request.args.get("backend", "demo")
        model = request.args.get("model") or None
        if backend != "demo" and not task:
            def err():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Scrivi prima un compito.'})}\n\n"
                yield "data: null\n\n"
            return Response(err(), mimetype="text/event-stream")
        return Response(
            _stream_agent(task, backend, model),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run_server(host: str = "127.0.0.1", port: int = 5001) -> None:
    _load_dotenv()
    create_app().run(host=host, port=port, threaded=True)


# Innova Web Design Studio brand: yellow #F2BF32 / black #121212 / white #FFFFFF.
INDEX_HTML = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Innova · Agente AI</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 55 51'%3E%3Cpath d='M0 0L25 15V50L0 49.9899V0Z' fill='%23F2BF32'/%3E%3Cpath d='M30 0.0100708L55 15.0101V50.0101L30 50V0.0100708Z' fill='%23F2BF32'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Host+Grotesk:wght@300;400;500;600;700&family=Poppins:wght@400;500;600&family=Archivo+Black&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#121212; --panel:#1A1A1A; --panel-2:#151515; --line:rgba(255,255,255,.09);
    --txt:#FFFFFF; --dim:#888888; --yellow:#F2BF32; --yellow-soft:rgba(242,191,50,.14);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font-family:'Host Grotesk','Inter',-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
    -webkit-font-smoothing:antialiased;}
  .wrap{max-width:840px;margin:0 auto;padding:48px 22px 72px}
  header{text-align:left}
  .logo svg{width:152px;height:auto;display:block}
  .eyebrow{margin:26px 0 0;font-weight:600;font-size:12px;letter-spacing:.18em;
    text-transform:uppercase;color:var(--yellow)}
  .title{margin:8px 0 0;font-family:'Archivo Black','Host Grotesk',system-ui;
    font-weight:400;font-size:clamp(28px,5vw,44px);text-transform:uppercase;
    letter-spacing:-.01em;line-height:1.02}
  .title .y{color:var(--yellow)}
  .sub{margin:14px 0 0;color:var(--dim);font-size:15px;max-width:560px;line-height:1.55}

  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:18px;margin-top:34px}
  textarea{width:100%;min-height:66px;resize:vertical;background:var(--panel-2);color:var(--txt);
    border:1px solid var(--line);border-radius:8px;padding:12px 13px;font-size:15px;
    font-family:inherit;outline:none;transition:border-color .15s}
  textarea::placeholder{color:#5c5c5c}
  textarea:focus{border-color:var(--yellow)}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
  .chip{font-size:12.5px;color:var(--dim);background:var(--panel-2);border:1px solid var(--line);
    border-radius:999px;padding:6px 12px;cursor:pointer;transition:.15s}
  .chip:hover{color:var(--txt);border-color:var(--yellow)}
  .row{display:flex;gap:12px;align-items:center;margin-top:16px;flex-wrap:wrap}
  select{font-family:'Poppins',system-ui;font-size:14px;background:var(--panel-2);color:var(--txt);
    border:1px solid var(--line);border-radius:8px;padding:10px 12px;outline:none;cursor:pointer}
  button{font-family:'Poppins',system-ui;font-size:14px;font-weight:600;border:none;
    border-radius:8px;padding:11px 20px;background:var(--yellow);color:#121212;cursor:pointer;
    transition:.15s}
  button:hover:not(:disabled){background:#f7cd54;transform:translateY(-1px)}
  button:disabled{opacity:.5;cursor:not-allowed}
  #status{color:var(--dim);font-size:13px;min-height:18px;margin-left:auto;font-weight:500}

  .trace{margin-top:26px;font-size:14px;line-height:1.55}
  .step{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:13px 15px;margin-bottom:11px;animation:fade .28s ease}
  @keyframes fade{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .lbl{font-weight:700;font-family:'Poppins',system-ui}
  .n{color:var(--dim);font-weight:700}
  .l-thought{color:var(--dim)} .v-thought{color:var(--txt)}
  .l-action{color:var(--yellow)} .v-action{color:var(--yellow)}
  .l-obs{color:var(--dim)} .v-obs{color:#cfcfcf}
  .note{border-left:3px solid var(--yellow);padding-left:11px;color:#cfcfcf}
  .note .lbl{color:var(--yellow)}
  .final{background:var(--yellow);color:#121212;border-radius:12px;padding:15px 16px;
    margin-top:4px;white-space:pre-wrap;font-weight:500}
  .final .lbl{color:#121212;font-family:'Poppins',system-ui}
  .errbox{border:1px solid var(--yellow);background:var(--yellow-soft);border-radius:8px;
    padding:12px 14px;color:#f0d27a}
  .pulse::after{content:"▋";margin-left:1px;animation:blink 1s steps(2) infinite;color:var(--yellow)}
  @keyframes blink{50%{opacity:0}}

  footer{margin-top:42px;padding-top:20px;border-top:1px solid var(--line);
    color:var(--dim);font-size:12.5px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  footer .cde{color:var(--txt);font-weight:600}
  footer a{color:var(--dim);text-decoration:none}
  footer a:hover{color:var(--yellow)}
  footer .sep{opacity:.4}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo"><svg width="285" height="93" viewBox="0 0 285 93" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M71.94 67.2438H75.8009L72.5734 88.5734H68.7427L67.3854 78.5485L66.028 88.5734H62.1973L59 67.2438H62.8307L64.1881 80.8033L65.5756 67.2438H69.1952L70.5827 80.8033L71.94 67.2438Z" fill="white"/>
<path d="M76.5161 88.5734V67.2438H84.6903V70.8089H80.7993V76.5679H83.6949V79.6454H80.7993V84.8864H84.6903V88.5734H76.5161Z" fill="white"/>
<path d="M85.7064 67.2438H90.3817C91.6787 67.2438 93.0663 67.7618 94.0013 68.6759C94.9364 69.6814 95.4793 70.9917 95.4491 72.3934V74.4044C95.4491 75.5928 94.876 76.7202 94.1521 77.6039C95.0269 78.5485 95.6905 79.7978 95.6905 81.108V83.4543C95.7206 84.856 95.1777 86.1967 94.2125 87.2022C93.3076 88.0859 91.9201 88.5734 90.623 88.5734H85.7064V67.2438ZM90.3817 70.7784H89.8991V76.2327H90.4421C90.985 76.2327 91.2565 75.867 91.2565 75.349V71.662C91.2565 71.0831 90.9247 70.7784 90.3817 70.7784ZM91.4978 84.1856V80.1634C91.5279 79.9501 91.4374 79.7064 91.2565 79.5235C91.1057 79.3407 90.8945 79.2493 90.6532 79.2493H89.8991V85.0997H90.623C91.2866 85.0997 91.4978 84.795 91.4978 84.1856Z" fill="white"/>
<path d="M111.764 68.6454C112.729 69.651 113.272 70.9917 113.242 72.3934V83.4543C113.272 84.856 112.729 86.1967 111.764 87.2022C110.829 88.0859 109.592 88.6039 108.326 88.5734H103.439V67.2438H108.326C109.592 67.2438 110.829 67.7618 111.764 68.6454ZM109.049 84.1856V71.662C109.049 71.0526 108.748 70.7479 108.326 70.7479H107.602V85.0997H108.326C108.808 85.0997 109.049 84.795 109.049 84.1856Z" fill="white"/>
<path d="M114.574 88.5734V67.2438H122.748V70.8089H118.857V76.5679H121.752V79.6454H118.857V84.8864H122.748V88.5734H114.574Z" fill="white"/>
<path d="M133.145 81.8698V84.3379C133.145 85.7091 132.692 86.8061 131.667 87.8116C130.852 88.6039 129.766 89 128.228 89C126.72 89 125.755 88.6343 124.94 87.8116C123.915 86.8061 123.462 85.7091 123.462 84.3379V80.9557H127.504V84.5512C127.504 85.1607 128.047 85.2216 128.228 85.2216C128.379 85.2216 128.922 85.1607 128.922 84.5512V81.9612C128.922 81.0471 128.68 80.9557 127.685 79.9806L124.608 77.1773C123.824 76.385 123.462 75.6233 123.462 74.313V71.6316C123.462 70.2909 123.915 69.1634 124.94 68.1579C125.755 67.3352 126.72 67 128.258 67C129.766 67 130.732 67.3352 131.546 68.1579C132.572 69.1634 133.024 70.2909 133.024 71.6316V74.7091H128.922V71.4488C128.922 70.8698 128.53 70.7479 128.319 70.7479C128.138 70.7479 127.957 70.8089 127.836 70.9307C127.715 71.0526 127.655 71.2355 127.685 71.4183V73.8864C127.685 74.892 128.228 75.0748 128.922 75.8061L132.029 78.4875C132.964 79.4321 133.145 80.2244 133.145 81.8698Z" fill="white"/>
<path d="M134.339 88.5734V67.2438H138.501V88.5734H134.339Z" fill="white"/>
<path d="M144.789 78.3352H149.675V88.5734H148.499C148.499 88.5734 148.348 87.5374 148.318 87.5983C147.413 88.5125 146.599 89 144.789 89C143.492 89 142.255 88.7258 141.38 87.8421C140.415 86.867 139.902 85.8006 139.902 84.3379V71.6316C139.902 70.1994 140.415 69.1025 141.38 68.1579C142.255 67.2438 143.492 67 144.789 67C146.086 67 147.292 67.2438 148.197 68.1579C149.162 69.1025 149.675 70.1994 149.675 71.6316V75.9889H145.513V71.4183C145.513 70.7784 145 70.7175 144.789 70.7175C144.578 70.7175 144.065 70.7784 144.065 71.4183V84.5817C144.065 85.2216 144.638 85.2825 144.849 85.2825C145.241 85.2825 145.603 85.0083 145.633 84.5817V80.7119H144.789V78.3352Z" fill="white"/>
<path d="M161.113 88.5734H157.343L154.749 77.0249V88.5734H151.099V67.2438H154.99L157.434 79.2493V67.2438H161.113V88.5734Z" fill="white"/>
<path d="M178.596 81.8698V84.3379C178.596 85.7091 178.143 86.8061 177.118 87.8116C176.303 88.6039 175.217 89 173.679 89C172.171 89 171.206 88.6343 170.391 87.8116C169.366 86.8061 168.913 85.7091 168.913 84.3379V80.9557H172.955V84.5512C172.955 85.1607 173.498 85.2216 173.679 85.2216C173.83 85.2216 174.373 85.1607 174.373 84.5512V81.9612C174.373 81.0471 174.131 80.9557 173.136 79.9806L170.059 77.1773C169.275 76.385 168.913 75.6233 168.913 74.313V71.6316C168.913 70.2909 169.366 69.1634 170.391 68.1579C171.206 67.3352 172.171 67 173.709 67C175.217 67 176.183 67.3352 176.997 68.1579C178.023 69.1634 178.475 70.2909 178.475 71.6316V74.7091H174.373V71.4488C174.373 70.8698 173.981 70.7479 173.77 70.7479C173.589 70.7479 173.408 70.8089 173.287 70.9307C173.166 71.0526 173.106 71.2355 173.136 71.4183V73.8864C173.136 74.892 173.679 75.0748 174.373 75.8061L177.48 78.4875C178.415 79.4321 178.596 80.2244 178.596 81.8698Z" fill="white"/>
<path d="M181.66 88.5734V70.8089H178.885V67.2438H188.658V70.8089H185.883V88.5734H181.66Z" fill="white"/>
<path d="M194.854 84.6122V67.2742H198.986V84.277C199.017 85.6177 198.474 86.8975 197.539 87.8421C196.604 88.7258 195.397 89 194.1 89C192.803 89 191.536 88.7258 190.661 87.8421C189.726 86.8975 189.183 85.6177 189.214 84.277V67.2742H193.346V84.6122C193.346 85.2216 193.889 85.2825 194.1 85.2825C194.311 85.2825 194.854 85.2216 194.854 84.6122Z" fill="white"/>
<path d="M208.764 68.6454C209.729 69.651 210.272 70.9917 210.242 72.3934V83.4543C210.272 84.856 209.729 86.1967 208.764 87.2022C207.828 88.0859 206.592 88.6039 205.325 88.5734H200.438V67.2438H205.325C206.592 67.2438 207.828 67.7618 208.764 68.6454ZM206.049 84.1856V71.662C206.049 71.0526 205.747 70.7479 205.325 70.7479H204.601V85.0997H205.325C205.808 85.0997 206.049 84.795 206.049 84.1856Z" fill="white"/>
<path d="M211.573 88.5734V67.2438H215.735V88.5734H211.573Z" fill="white"/>
<path d="M225.522 68.1579C226.487 69.1025 227 70.1994 227 71.6011V84.3684C227 85.831 226.487 86.8975 225.522 87.8421C224.587 88.7562 223.35 89 222.053 89C220.756 89 219.489 88.7562 218.584 87.8421C217.649 86.8975 217.137 85.831 217.137 84.3684V71.6011C217.137 70.1994 217.649 69.1025 218.584 68.1579C219.489 67.2438 220.756 67 222.053 67C223.35 67 224.587 67.2438 225.522 68.1579ZM222.837 84.6122V71.3878C222.837 70.7784 222.295 70.7175 222.053 70.7175C221.842 70.7175 221.299 70.7784 221.299 71.3878V84.6122C221.299 85.2216 221.842 85.313 222.053 85.313C222.295 85.313 222.837 85.2216 222.837 84.6122Z" fill="white"/>
<path fill-rule="evenodd" clip-rule="evenodd" d="M1 6.11697L26.1779 21.2237V56.4729L1 56.4627V6.11697Z" fill="white"/>
<path d="M57.1777 21.1171L57.1787 41.2489L57.1777 56.3661L32 56.3554V6.00967L57.1777 21.1171ZM82.3564 6.00967V56.3554L57.1787 41.2489V5.99991L82.3564 6.00967Z" fill="white"/>
<path d="M136.961 6.12686V56.4726L111.783 41.3661V56.4833L86.6045 56.4726V6.12686L111.783 21.2343V6.1171L136.961 6.12686Z" fill="white"/>
<path d="M197.388 30.7913C197.388 46.0873 184.988 58.4871 169.692 58.4871C154.396 58.4871 141.997 46.0873 141.997 30.7913C141.997 15.4954 154.396 3.09561 169.692 3.09561C184.988 3.09561 197.388 15.4954 197.388 30.7913Z" fill="white"/>
<path d="M250.765 6.11697L223.07 56.4729L195.374 6.11698L219.545 6.11699L223.07 12.1597L226.595 6.11699L250.765 6.11697Z" fill="white"/>
<path d="M228.609 56.4728L256.304 6.11696L284 56.4728L259.829 56.4728L256.304 50.4301L252.779 56.4728L228.609 56.4728Z" fill="white"/>
</svg></div>
    <p class="eyebrow">Agente AI · Demo live</p>
    <h1 class="title">Pensa <span class="y">·</span> Agisce <span class="y">·</span> Osserva</h1>
    <p class="sub">Guarda un agente AI ragionare in tempo reale e usare gli strumenti per arrivare alla risposta, un passo alla volta.</p>
  </header>

  <div class="card">
    <textarea id="task" placeholder="Scrivi un compito…  es. Quanto fa 18 × 24 e qual è la capitale del Giappone?"></textarea>
    <div class="chips" id="chips"></div>
    <div class="row">
      <select id="backend" aria-label="Motore">
        <option value="demo">Demo (offline, senza chiave)</option>
        <option value="anthropic">Claude (live)</option>
        <option value="openai">OpenAI (live)</option>
      </select>
      <button id="run">Avvia l'agente</button>
      <span id="status"></span>
    </div>
  </div>

  <div class="trace" id="trace"></div>

  <footer>
    <span class="cde">Creating Digital Experiences</span>
    <span class="sep">·</span>
    <a href="https://www.innovadesignstudio.it" target="_blank" rel="noopener">innovadesignstudio.it</a>
  </footer>
</div>

<script>
const $ = s => document.querySelector(s);
const trace = $("#trace"), statusEl = $("#status"), runBtn = $("#run");
let es = null;

const examples = [
  "Quanto fa 18 × 24 e qual è la capitale del Giappone?",
  "Cosa fa Innova? E conta le parole in 'Creating Digital Experiences'.",
  "Trova la capitale d'Italia, poi calcola 144 × 12.",
];
const chips = $("#chips");
examples.forEach(t => {
  const c = document.createElement("span");
  c.className = "chip"; c.textContent = t;
  c.onclick = () => { $("#task").value = t; $("#backend").value = "anthropic"; };
  chips.appendChild(c);
});

function el(cls, html){ const d = document.createElement("div"); d.className = cls; d.innerHTML = html; return d; }
function esc(s){ return (s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function addStep(s){
  const box = document.createElement("div");
  box.className = "step";
  if(s.thought) box.appendChild(el("",
    `<span class="n">${s.index}.</span> <span class="lbl l-thought">Pensiero:</span> <span class="v-thought">${esc(s.thought)}</span>`));
  if(s.action){
    box.appendChild(el("mono",
      `&nbsp;&nbsp;&nbsp;<span class="lbl l-action">Azione:</span> <span class="v-action">${esc(s.action)}(${esc(JSON.stringify(s.action_input))})</span>`));
    box.appendChild(el("mono",
      `&nbsp;&nbsp;&nbsp;<span class="lbl l-obs">Osservazione:</span> <span class="v-obs">${esc(s.observation)}</span>`));
  } else if(s.observation){
    box.appendChild(el("note", `<span class="lbl">Nota:</span> ${esc(s.observation)}`));
  }
  trace.appendChild(box);
  window.scrollTo(0, document.body.scrollHeight);
}

function stop(){ if(es){ es.close(); es = null; } runBtn.disabled = false; statusEl.classList.remove("pulse"); }

runBtn.onclick = () => {
  stop();
  trace.innerHTML = "";
  const task = $("#task").value.trim();
  const backend = $("#backend").value;
  runBtn.disabled = true;
  statusEl.textContent = "sto ragionando ";
  statusEl.classList.add("pulse");

  const url = `/run?backend=${encodeURIComponent(backend)}&task=${encodeURIComponent(task)}`;
  es = new EventSource(url);
  es.onmessage = (ev) => {
    if(ev.data === "null"){ stop(); return; }
    const m = JSON.parse(ev.data);
    if(m.type === "start"){ statusEl.textContent = `in esecuzione · ${m.backend}:${m.model} `; statusEl.classList.add("pulse"); }
    else if(m.type === "step"){ addStep(m); }
    else if(m.type === "done"){
      if(m.final_answer != null){
        trace.appendChild(el("final", `<span class="lbl">Risposta finale:</span> ${esc(m.final_answer)}`));
      } else {
        trace.appendChild(el("errbox", `Interrotto: ${esc(m.stopped_reason)} (nessuna risposta finale)`));
      }
      statusEl.textContent = m.succeeded ? "fatto ✓" : "interrotto";
      statusEl.classList.remove("pulse"); stop();
    }
    else if(m.type === "error"){
      trace.appendChild(el("errbox", `Errore: ${esc(m.message)}`));
      statusEl.textContent = "errore"; statusEl.classList.remove("pulse"); stop();
    }
  };
  es.onerror = () => { statusEl.textContent = "connessione chiusa"; stop(); };
};
</script>
</body>
</html>
"""
