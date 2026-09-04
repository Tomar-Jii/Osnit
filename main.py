import asyncio
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")
INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

IDX_PHONE = "idx_phone"
IDX_AADHAR = "idx_aadhar"

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ── Dedup & Connected Records ───────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out


# ── Search Logic ────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        elif field == "otherNumber":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        if field == "name":
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8

    if is_num:
        all_rows = []
        searched = []
        if _idx_ready("phone"):
            r = _run_field_search("phoneNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
        if not all_rows and _idx_ready("aadhar"):
            r = _run_field_search("aadharNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {
            "query": q, "searched_fields": searched,
            "count": len(all_rows), "results": all_rows,
        }
    else:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}


# ── Modern Web Dashboard Template ──────────────────────────────────────────
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
  <title>KAL-X Search Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card: #121827;
      --card-border: #1f293d;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --text: #f8fafc;
      --text-dim: #94a3b8;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    
    header {
      padding: 14px 20px;
      background: rgba(18, 24, 39, 0.9);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.75rem;
      color: #34d399;
    }
    .status-dot { width: 7px; height: 7px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; }
    .nav-links a { color: var(--text-dim); text-decoration: none; font-size: 0.85rem; margin-left: 14px; transition: 0.2s; }
    .nav-links a:hover { color: var(--text); }

    .main-container { max-width: 860px; margin: 0 auto; width: 100%; padding: 30px 16px; flex: 1; }
    .hero { text-align: center; margin-bottom: 28px; }
    .hero h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.5px; }
    .hero p { color: var(--text-dim); font-size: 0.95rem; }

    .search-box {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      margin-bottom: 24px;
    }
    .input-row { display: flex; gap: 10px; }
    .search-input {
      flex: 1;
      background: #070a12;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 13px 16px;
      color: #fff;
      font-size: 1rem;
      outline: none;
      transition: 0.2s;
    }
    .search-input:focus { border-color: var(--accent); box-shadow: 0 0 12px var(--accent-glow); }
    .limit-select {
      background: #070a12;
      border: 1px solid var(--card-border);
      color: var(--text-dim);
      border-radius: 12px;
      padding: 0 12px;
      font-size: 0.9rem;
      outline: none;
    }
    .search-btn {
      background: linear-gradient(135deg, var(--accent), var(--accent-hover));
      border: none;
      color: #fff;
      padding: 13px 24px;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.95rem;
      transition: 0.2s;
    }
    .search-btn:active { transform: scale(0.97); }

    .stats-bar {
      display: flex;
      justify-content: space-between;
      color: var(--text-dim);
      font-size: 0.85rem;
      margin-bottom: 16px;
      padding: 0 4px;
    }

    .results-grid { display: flex; flex-direction: column; gap: 14px; }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px;
      transition: 0.2s;
      position: relative;
    }
    .card:hover { border-color: #2e3d5b; transform: translateY(-2px); }
    .card-title { font-size: 1.15rem; font-weight: 600; color: #fff; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .copy-btn {
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--card-border);
      color: var(--text-dim);
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
    }
    .copy-btn:hover { color: #fff; border-color: var(--accent); }
    
    .data-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px 16px; font-size: 0.88rem; }
    .data-item { display: flex; flex-direction: column; }
    .data-item label { color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .data-item span { color: #e2e8f0; word-break: break-word; font-weight: 500; }

    .badge-chip {
      display: inline-block;
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.78rem;
      margin-right: 6px;
      margin-top: 4px;
    }

    .loader { text-align: center; padding: 40px; color: var(--text-dim); display: none; }
    .spinner {
      width: 36px; height: 36px; border: 3px solid rgba(255,255,255,0.1); border-top-color: var(--accent);
      border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      text-align: center;
      padding: 22px;
      color: #64748b;
      font-size: 0.85rem;
      border-top: 1px solid var(--card-border);
      background: #080b13;
    }
    footer b { color: #cbd5e1; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h2 style="font-size: 1.1rem; letter-spacing: 0.5px;">KAL-X <span style="color:var(--accent);">SEARCH</span></h2>
      <div class="status-badge"><div class="status-dot"></div> Remote Ready</div>
    </div>
    <div class="nav-links">
      <a href="/docs">Swagger Docs</a>
      <a href="/health">Health</a>
    </div>
  </header>

  <div class="main-container">
    <div class="hero">
      <h1>Intelligence Database Lookup</h1>
      <p>Search over <b>2.5 Billion</b> records across phone numbers and identity indexes.</p>
    </div>

    <div class="search-box">
      <div class="input-row">
        <input type="text" id="searchInput" class="search-input" placeholder="Enter Phone Number or Search Query..." autofocus onkeydown="if(event.key==='Enter') executeSearch()">
        <select id="limitSelect" class="limit-select">
          <option value="5">5 Results</option>
          <option value="10" selected>10 Results</option>
          <option value="25">25 Results</option>
        </select>
        <button class="search-btn" onclick="executeSearch()">🔍 Search</button>
      </div>
    </div>

    <div id="statsBar" class="stats-bar" style="display: none;">
      <span id="statsQuery">Query: -</span>
      <span id="statsCount">Found: 0</span>
    </div>

    <div id="loader" class="loader">
      <div class="spinner"></div>
      Searching distributed parquet index...
    </div>

    <div id="resultsGrid" class="results-grid"></div>
  </div>

  <footer>
    👨‍💻 Developed by <b>Tomar Ji</b> | Powered by DuckDB & FastAPI
  </footer>

  <script>
    async function executeSearch() {
      const q = document.getElementById('searchInput').value.trim();
      const limit = document.getElementById('limitSelect').value;
      if (!q) return;

      const loader = document.getElementById('loader');
      const resultsGrid = document.getElementById('resultsGrid');
      const statsBar = document.getElementById('statsBar');
      const statsQuery = document.getElementById('statsQuery');
      const statsCount = document.getElementById('statsCount');

      resultsGrid.innerHTML = '';
      loader.style.display = 'block';
      statsBar.style.display = 'none';

      try {
        const res = await fetch(`/search?q=${encodeURIComponent(q)}&limit=${limit}&pretty=true`);
        const data = await res.json();
        loader.style.display = 'none';

        statsBar.style.display = 'flex';
        statsQuery.innerText = `Query: ${q}`;
        statsCount.innerText = `Found: ${data.count || 0} results`;

        if (!data.results || data.results.length === 0) {
          resultsGrid.innerHTML = `
            <div class="card" style="text-align: center; color: var(--text-dim); padding: 30px;">
              ❌ Koi record nahi mila is query ke liye.
            </div>`;
          return;
        }

        data.results.forEach((item, index) => {
          const card = document.createElement('div');
          card.className = 'card';

          let connectedHtml = '';
          if (item.connected_numbers && item.connected_numbers.length > 0) {
            connectedHtml = `
              <div style="margin-top: 12px;">
                <label style="color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase;">Connected Numbers</label>
                <div>${item.connected_numbers.map(c => `<span class="badge-chip">${c.field}: ${c.value}</span>`).join('')}</div>
              </div>`;
          }

          card.innerHTML = `
            <div class="card-title">
              <span>#${index + 1} ${item.name || 'Unknown Name'}</span>
              <button class="copy-btn" onclick="copyCardData(this, ${JSON.stringify(JSON.stringify(item))})">📋 Copy</button>
            </div>
            <div class="data-grid">
              ${item.fathersName ? `<div class="data-item"><label>Father's Name</label><span>${item.fathersName}</span></div>` : ''}
              ${item.phoneNumber ? `<div class="data-item"><label>Phone Number</label><span style="color:#38bdf8;">${item.phoneNumber}</span></div>` : ''}
              ${item.aadharNumber ? `<div class="data-item"><label>Aadhaar</label><span>${item.aadharNumber}</span></div>` : ''}
              ${item.address ? `<div class="data-item"><label>Address</label><span>${item.address}</span></div>` : ''}
              ${item.district ? `<div class="data-item"><label>District</label><span>${item.district}</span></div>` : ''}
              ${item.state ? `<div class="data-item"><label>State</label><span>${item.state}</span></div>` : ''}
              ${item.pincode ? `<div class="data-item"><label>Pincode</label><span>${item.pincode}</span></div>` : ''}
              ${item.source ? `<div class="data-item"><label>Source</label><span>${item.source}</span></div>` : ''}
            </div>
            ${connectedHtml}
          `;
          resultsGrid.appendChild(card);
        });

      } catch (err) {
        loader.style.display = 'none';
        resultsGrid.innerHTML = `
          <div class="card" style="text-align: center; color: #ef4444; padding: 30px;">
            ⚠️ Query fail ho gayi ya server response time out ho gaya. Thodi der baad try karein.
          </div>`;
      }
    }

    function copyCardData(btn, jsonStr) {
      const data = JSON.parse(jsonStr);
      let text = '';
      for (const [k, v] of Object.entries(data)) {
        if (v && typeof v !== 'object') text += `${k}: ${v}\\n`;
      }
      navigator.clipboard.writeText(text);
      const old = btn.innerText;
      btn.innerText = '✅ Copied!';
      setTimeout(() => btn.innerText = old, 1500);
    }
  </script>
</body>
</html>
"""

# ── FastAPI (for API & UI access) ───────────────────────────────────────────
fastapi_app = FastAPI(title="Search API")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@fastapi_app.get("/", response_class=HTMLResponse)
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return Response(
            content=json.dumps({
                "app": "Search API",
                "records": 2_504_793_870,
                "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
                "index_source": INDEX_SOURCE,
                "columns": SEARCH_FIELDS,
                "docs": "/docs",
                "developer": "Tomar Ji",
            }, indent=2),
            media_type="application/json"
        )
    return HTML_DASHBOARD


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "raw_database_required": False,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE,
    }


@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    result = {
        "success": bool(data["count"]),
        **data,
        "number": q_val,
        "total": data["count"],
    }
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool,
            _run_field_search,
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps(
            {"searches": len(req.queries), "results": list(results)},
            indent=2,
            ensure_ascii=False,
        ),
        media_type="application/json",
    )


# ── Pinger (keeps app alive) ──────────────────────────────────────────────
async def pinger():
    """Ping the /health endpoint every 2 minutes to prevent idle shutdown."""
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    print(f"[Pinger] OK at {asyncio.get_event_loop().time()}")
                else:
                    print(f"[Pinger] Unexpected status: {resp.status_code}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")


@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())


# ── Gradio UI Fallback (Available at /gradio) ───────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)


def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Search query khali hai."
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n❌ **No data found**."
    header = f"🔍 **Query:** `{q}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = []
    for i, row in enumerate(results, 1):
        parts.append(f"### Result {i}\n{format_result(row)}")
    return header + "\n\n---\n\n".join(parts)


def build_ui():
    with gr.Blocks(title="Search Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Search Dashboard")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search Query", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---\n**Developer:** Tomar Ji")
    return demo


demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
