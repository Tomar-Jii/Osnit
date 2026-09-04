import asyncio
import glob
import json
import os
import re
import threading
import time
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
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "4"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

# Blacklist numbers (Inka data search me kabhi show nahi hoga)
BLACKLISTED_NUMBERS = {
    "9926888306",
}

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

# ── Fast In-Memory Cache & Metrics ──────────────────────────────────────────
QUERY_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 3600
METRICS = {
    "total_searches": 0,
    "cache_hits": 0,
    "start_time": time.time()
}

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="kalx")


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


# ── Clean & Helper Functions ────────────────────────────────────────────────
def _clean_number(val: str) -> str:
    clean = re.sub(r"\D", "", val.strip())
    if len(clean) == 12 and clean.startswith("91"):
        clean = clean[2:]
    elif len(clean) == 11 and clean.startswith("0"):
        clean = clean[1:]
    return clean


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


# ── Search Engine Logic ─────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    
    # Blacklist check
    clean_val = _clean_number(value)
    if clean_val in BLACKLISTED_NUMBERS or value.strip() in BLACKLISTED_NUMBERS:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}

    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        else:
            return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    else:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    
    # Filter blacklisted numbers from connected results
    filtered_results = []
    for row in results:
        ph = _clean_number(str(row.get("phoneNumber") or ""))
        if ph in BLACKLISTED_NUMBERS:
            continue
        filtered_results.append(row)

    return {"field": field, "value": value, "mode": mode, "count": len(filtered_results), "results": filtered_results}


def _unified_search(q: str, limit: int = 10) -> dict:
    raw_q = q.strip()
    clean_q = _clean_number(raw_q)
    target = clean_q if clean_q else raw_q

    # Blacklist check (Instant empty response)
    if clean_q in BLACKLISTED_NUMBERS or raw_q in BLACKLISTED_NUMBERS:
        return {"query": target, "searched_fields": [], "count": 0, "results": [], "cached": False}

    # Check In-Memory Cache
    now = time.time()
    if target in QUERY_CACHE:
        cache_time, cached_data = QUERY_CACHE[target]
        if now - cache_time < CACHE_TTL:
            METRICS["cache_hits"] += 1
            return cached_data

    METRICS["total_searches"] += 1
    is_num = target.isdigit() and len(target) >= 8

    if is_num:
        all_rows = []
        searched = []
        if _idx_ready("phone"):
            r = _run_field_search("phoneNumber", target, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
        if not all_rows and _idx_ready("aadhar"):
            r = _run_field_search("aadharNumber", target, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")

        # Double check output rows
        safe_rows = []
        for r in all_rows:
            ph = _clean_number(str(r.get("phoneNumber") or ""))
            if ph not in BLACKLISTED_NUMBERS:
                safe_rows.append(r)

        safe_rows = _cap_duplicates(safe_rows)[:limit]
        res = {
            "query": target,
            "searched_fields": searched,
            "count": len(safe_rows),
            "results": safe_rows,
            "cached": False,
        }
        QUERY_CACHE[target] = (now, {**res, "cached": True})
        return res

    return {"query": target, "searched_fields": [], "count": 0, "results": [], "cached": False}


# ── Web Dashboard Template ──────────────────────────────────────────────────
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KAL-X Search Intelligence</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #070a12;
      --card: #111726;
      --border: #1e293b;
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --text: #f8fafc;
      --dim: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    header { padding: 14px 24px; background: rgba(17, 23, 38, 0.8); backdrop-filter: blur(10px); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 50; }
    .badge { display: flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); color: #34d399; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; }
    .dot { width: 7px; height: 7px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; }
    .nav a { color: var(--dim); text-decoration: none; font-size: 0.85rem; margin-left: 16px; transition: 0.2s; }
    .nav a:hover { color: #fff; }
    .main { max-width: 860px; margin: 0 auto; width: 100%; padding: 32px 16px; flex: 1; }
    .hero { text-align: center; margin-bottom: 28px; }
    .hero h1 { font-size: 2rem; font-weight: 700; margin-bottom: 8px; }
    .hero p { color: var(--dim); font-size: 0.95rem; }
    .search-panel { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); margin-bottom: 24px; }
    .search-row { display: flex; gap: 10px; }
    .input-box { flex: 1; background: #0b0f19; border: 1px solid var(--border); border-radius: 12px; padding: 13px 16px; color: #fff; font-size: 1rem; outline: none; }
    .input-box:focus { border-color: var(--accent); box-shadow: 0 0 10px var(--accent-glow); }
    .btn { background: var(--accent); border: none; color: #fff; padding: 13px 24px; border-radius: 12px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    .btn:active { transform: scale(0.96); }
    .status-bar { display: flex; justify-content: space-between; color: var(--dim); font-size: 0.85rem; margin-bottom: 16px; padding: 0 4px; }
    .grid { display: flex; flex-direction: column; gap: 14px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; transition: 0.2s; }
    .card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .card-title { font-size: 1.15rem; font-weight: 600; color: #fff; }
    .copy-btn { background: rgba(255,255,255,0.06); border: 1px solid var(--border); color: var(--dim); font-size: 0.75rem; padding: 4px 10px; border-radius: 6px; cursor: pointer; }
    .fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px 16px; font-size: 0.88rem; }
    .item label { color: var(--dim); font-size: 0.72rem; text-transform: uppercase; display: block; margin-bottom: 2px; }
    .item span { color: #e2e8f0; font-weight: 500; word-break: break-word; }
    .badge-chip { display: inline-block; background: rgba(99, 102, 241, 0.15); color: #a5b4fc; padding: 3px 8px; border-radius: 6px; font-size: 0.78rem; margin: 4px 6px 0 0; }
    .loader { text-align: center; padding: 40px; color: var(--dim); display: none; }
    .spinner { width: 34px; height: 34px; border: 3px solid rgba(255,255,255,0.1); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 12px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    footer { text-align: center; padding: 22px; color: #64748b; font-size: 0.85rem; border-top: 1px solid var(--border); }
    footer b { color: #cbd5e1; }
  </style>
</head>
<body>
  <header>
    <div style="display:flex; align-items:center; gap:10px;">
      <h2 style="font-size:1.1rem;">KAL-X <span style="color:var(--accent);">SEARCH</span></h2>
      <div class="badge"><div class="dot"></div> Online</div>
    </div>
    <div class="nav">
      <a href="/docs">Docs</a>
      <a href="/metrics">Metrics</a>
      <a href="/health">Health</a>
    </div>
  </header>

  <div class="main">
    <div class="hero">
      <h1>Intelligence Database Lookup</h1>
      <p>Search over <b>2.5 Billion</b> records across phone and identity indexes.</p>
    </div>

    <div class="search-panel">
      <div class="search-row">
        <input type="text" id="qInput" class="input-box" placeholder="Phone number ya identifier enter karein..." onkeydown="if(event.key==='Enter') runLookup()">
        <button class="btn" onclick="runLookup()">🔍 Search</button>
      </div>
    </div>

    <div id="statusInfo" class="status-bar" style="display:none;">
      <span id="labelQuery"></span>
      <span id="labelFound"></span>
    </div>

    <div id="loader" class="loader">
      <div class="spinner"></div>
      Searching distributed parquet index...
    </div>

    <div id="outputGrid" class="grid"></div>
  </div>

  <footer>
    👨‍💻 Developed by <b>Tomar Ji</b> | Distributed Parquet Engine
  </footer>

  <script>
    async function runLookup() {
      const val = document.getElementById('qInput').value.trim();
      if (!val) return;

      const loader = document.getElementById('loader');
      const grid = document.getElementById('outputGrid');
      const statusBar = document.getElementById('statusInfo');
      const labelQuery = document.getElementById('labelQuery');
      const labelFound = document.getElementById('labelFound');

      grid.innerHTML = '';
      loader.style.display = 'block';
      statusBar.style.display = 'none';

      try {
        const res = await fetch(`/search?q=${encodeURIComponent(val)}&limit=10`);
        const data = await res.json();
        loader.style.display = 'none';

        statusBar.style.display = 'flex';
        labelQuery.innerText = `Query: ${data.number || val} ${data.cached ? '⚡ (Cached)' : ''}`;
        labelFound.innerText = `Found: ${data.total || 0} records`;

        if (!data.results || data.results.length === 0) {
          grid.innerHTML = `<div class="card" style="text-align:center; color:var(--dim);">❌ No records found for this query.</div>`;
          return;
        }

        data.results.forEach((row, i) => {
          const card = document.createElement('div');
          card.className = 'card';

          let connHtml = '';
          if (row.connected_numbers && row.connected_numbers.length > 0) {
            connHtml = `<div style="margin-top:12px;"><label style="color:var(--dim); font-size:0.72rem; text-transform:uppercase;">Connected</label><div>${row.connected_numbers.map(c => `<span class="badge-chip">${c.field}: ${c.value}</span>`).join('')}</div></div>`;
          }

          card.innerHTML = `
            <div class="card-top">
              <span class="card-title">#${i+1} ${row.name || 'Unknown Name'}</span>
              <button class="copy-btn" onclick="copyRecord(this, ${JSON.stringify(JSON.stringify(row))})">📋 Copy</button>
            </div>
            <div class="fields">
              ${row.fathersName ? `<div class="item"><label>Father</label><span>${row.fathersName}</span></div>` : ''}
              ${row.phoneNumber ? `<div class="item"><label>Phone</label><span style="color:#38bdf8;">${row.phoneNumber}</span></div>` : ''}
              ${row.address ? `<div class="item"><label>Address</label><span>${row.address}</span></div>` : ''}
              ${row.district ? `<div class="item"><label>District</label><span>${row.district}</span></div>` : ''}
              ${row.state ? `<div class="item"><label>State</label><span>${row.state}</span></div>` : ''}
              ${row.pincode ? `<div class="item"><label>Pincode</label><span>${row.pincode}</span></div>` : ''}
              ${row.source ? `<div class="item"><label>Source</label><span>${row.source}</span></div>` : ''}
            </div>
            ${connHtml}
          `;
          grid.appendChild(card);
        });
      } catch (err) {
        loader.style.display = 'none';
        grid.innerHTML = `<div class="card" style="text-align:center; color:#ef4444;">⚠️ Query timeout. Dubara try karein.</div>`;
      }
    }

    function copyRecord(btn, jsonStr) {
      const data = JSON.parse(jsonStr);
      let text = '';
      for (const [k, v] of Object.entries(data)) {
        if (v && typeof v !== 'object') text += `${k}: ${v}\\n`;
      }
      navigator.clipboard.writeText(text);
      btn.innerText = '✅ Copied!';
      setTimeout(() => btn.innerText = '📋 Copy', 1500);
    }
  </script>
</body>
</html>
"""

# ── FastAPI Routes ──────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="KAL-X Search API", version="2.5.0")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


class BulkScanRequest(BaseModel):
    numbers: list[str]
    limit_per_query: int = 5


@fastapi_app.get("/", response_class=HTMLResponse)
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return Response(
            content=json.dumps({
                "app": "KAL-X Search Intelligence API",
                "version": "2.5.0",
                "records": 2_504_793_870,
                "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
                "developer": "Tomar Ji",
                "endpoints": ["/search", "/search/bulk", "/export", "/metrics", "/health", "/docs"]
            }, indent=2),
            media_type="application/json"
        )
    return HTML_DASHBOARD


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "developer": "Tomar Ji",
    }


@fastapi_app.get("/metrics")
def metrics():
    uptime = round(time.time() - METRICS["start_time"], 2)
    return {
        "uptime_seconds": uptime,
        "total_searches": METRICS["total_searches"],
        "cache_hits": METRICS["cache_hits"],
        "cached_queries_count": len(QUERY_CACHE),
        "developer": "Tomar Ji"
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
        raise HTTPException(422, "Provide query via ?q= or ?mobile=")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)

    result = {
        "success": bool(data.get("count", 0)),
        **data,
        "number": q_val,
        "total": data.get("count", 0),
        "developer": "Tomar Ji"
    }
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@fastapi_app.post("/search/bulk")
async def bulk_search(req: BulkScanRequest):
    if not req.numbers:
        raise HTTPException(400, "Numbers list cannot be empty.")
    clean_list = req.numbers[:20]

    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, _unified_search, num, req.limit_per_query)
        for num in clean_list
    ]
    results = await asyncio.gather(*tasks)
    return {
        "developer": "Tomar Ji",
        "scanned_count": len(clean_list),
        "results": {num: res for num, res in zip(clean_list, results)}
    }


@fastapi_app.get("/export")
async def export_data(q: str = Query(...)):
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _unified_search, q, 50)
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False)
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=export_{_clean_number(q)}.json"}
    )


# ── Background Keep-Alive ───────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                await client.get(url)
            except Exception:
                pass


@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())


app = fastapi_app
