#!/usr/bin/env python3
"""Web dashboard for Auto Claude Code — read-only observer + feedback submission.

Uses only Python stdlib (http.server). Safe to run alongside the orchestrator.

Usage:
    python3 dashboard.py --config config.yaml --port 8505
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PORT = 8505
MAX_LOG_LINES = 500
MAX_LOC_HASHES_PER_REQUEST = 20
MAX_FEEDBACK_CONTENT_SIZE = 50 * 1024  # 50 KB
FEEDBACK_FILENAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]{0,98}\.(md|txt)$')

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section 1: Embedded SPA (HTML / CSS / JS)
# ---------------------------------------------------------------------------
STATIC_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto Claude Code Dashboard</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--card:#161b22;--border:#30363d;
  --text:#c9d1d9;--muted:#8b949e;
  --success:#3fb950;--fail:#f85149;--warn:#d29922;--info:#58a6ff;
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}
a{color:var(--info);text-decoration:none}
button{cursor:pointer;font-family:inherit;font-size:inherit}

/* Header */
.header{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid var(--border);background:var(--card)}
.header h1{font-size:18px;font-weight:600}
.header-right{display:flex;align-items:center;gap:16px;font-size:13px;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}
.dot.on{background:var(--success)}.dot.off{background:var(--fail)}
.toggle-btn{background:none;border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:4px;font-size:12px}
.toggle-btn.active{border-color:var(--info);color:var(--info)}

/* Container */
.container{max-width:1400px;margin:0 auto;padding:16px 24px}

/* Status cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}
.card-label{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:6px}
.card-value{font-size:24px;font-weight:600}
.card-value.success{color:var(--success)}.card-value.fail{color:var(--fail)}.card-value.warn{color:var(--warn)}
.bar-track{height:6px;background:var(--border);border-radius:3px;margin-top:8px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .3s}
.bar-fill.green{background:var(--success)}.bar-fill.red{background:var(--fail)}.bar-fill.yellow{background:var(--warn)}.bar-fill.blue{background:var(--info)}
.donut-container{display:flex;align-items:center;gap:12px}
.donut{width:48px;height:48px;border-radius:50%;position:relative}
.donut-hole{width:32px;height:32px;border-radius:50%;background:var(--card);position:absolute;top:8px;left:8px}

/* Filter bar */
.filter-bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.filter-bar select,.filter-bar input{background:var(--card);border:1px solid var(--border);color:var(--text);padding:6px 10px;border-radius:4px;font-size:13px}
.filter-bar input{flex:1;min-width:200px}

/* Table */
.table-wrap{overflow-x:auto;margin-bottom:20px}
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:8px 12px;border-bottom:2px solid var(--border);font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:var(--text)}
th .sort-arrow{margin-left:4px;font-size:10px}
td{padding:8px 12px;border-bottom:1px solid var(--border);vertical-align:top}
tr.success-row{border-left:3px solid var(--success)}
tr.fail-row{border-left:3px solid var(--fail)}
tr.expandable{cursor:pointer}
tr.expandable:hover{background:rgba(88,166,255,.05)}
.detail-row td{padding:16px 24px;background:rgba(22,27,34,.7)}
.detail-row pre{background:var(--bg);padding:12px;border-radius:6px;overflow-x:auto;font-size:12px;margin-top:8px;white-space:pre-wrap;word-break:break-word}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:500}
.badge-feedback{background:rgba(88,166,255,.15);color:var(--info)}
.badge-test{background:rgba(63,185,80,.15);color:var(--success)}
.badge-lint{background:rgba(210,153,34,.15);color:var(--warn)}
.badge-todo{background:rgba(139,148,158,.15);color:var(--muted)}
.badge-quality{background:rgba(248,81,73,.15);color:var(--fail)}
.badge-coverage{background:rgba(188,140,255,.15);color:#bc8cff}
.badge-claude_ideas{background:rgba(88,166,255,.15);color:var(--info)}
.badge-unknown{background:rgba(139,148,158,.15);color:var(--muted)}
.status-icon{font-size:16px}
.pagination{display:flex;align-items:center;gap:12px;justify-content:center;margin-top:12px;color:var(--muted);font-size:13px}
.pagination button{background:var(--card);border:1px solid var(--border);color:var(--text);padding:4px 12px;border-radius:4px}
.pagination button:disabled{opacity:.4;cursor:not-allowed}

/* Tabs */
.tabs{display:flex;border-bottom:1px solid var(--border);margin-bottom:16px}
.tab{padding:8px 16px;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;font-size:13px}
.tab:hover{color:var(--text)}.tab.active{color:var(--info);border-bottom-color:var(--info)}
.tab-count{background:var(--border);color:var(--text);padding:1px 6px;border-radius:10px;font-size:11px;margin-left:6px}

/* Feedback */
.feedback-form{display:flex;flex-direction:column;gap:8px;margin-bottom:16px;padding:16px;background:var(--card);border:1px solid var(--border);border-radius:8px}
.feedback-form input,.feedback-form textarea{background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:4px;font-family:inherit;font-size:13px}
.feedback-form textarea{min-height:100px;resize:vertical}
.feedback-form button{align-self:flex-start;background:var(--info);color:#fff;border:none;padding:8px 20px;border-radius:4px;font-weight:500}
.feedback-form button:hover{opacity:.9}
.feedback-item{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:8px}
.feedback-item-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.feedback-item-name{font-weight:500;font-size:13px}
.feedback-item-content{color:var(--muted);font-size:13px;white-space:pre-wrap;word-break:break-word}
.feedback-item .delete-btn{background:none;border:1px solid var(--border);color:var(--fail);padding:2px 8px;border-radius:4px;font-size:11px}
.feedback-item .expand-btn{background:none;border:none;color:var(--info);padding:0;font-size:12px;cursor:pointer}

/* Log viewer */
.log-section{margin-top:20px}
.log-header{display:flex;justify-content:space-between;align-items:center;cursor:pointer;padding:10px 16px;background:var(--card);border:1px solid var(--border);border-radius:8px 8px 0 0}
.log-header h3{font-size:14px}
.log-header .refresh-btn{background:none;border:1px solid var(--border);color:var(--text);padding:2px 10px;border-radius:4px;font-size:12px}
.log-panel{background:var(--bg);border:1px solid var(--border);border-top:none;border-radius:0 0 8px 8px;padding:12px;max-height:400px;overflow-y:auto;font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;font-size:12px;line-height:1.6}
.log-panel.collapsed{display:none}
.log-line{white-space:pre-wrap;word-break:break-all}
.log-line.error{color:var(--fail)}.log-line.warning{color:var(--warn)}.log-line.info{color:var(--text)}

/* Misc */
.empty-state{text-align:center;padding:40px;color:var(--muted)}
.section-title{font-size:16px;font-weight:600;margin-bottom:12px}
.msg{padding:8px 12px;border-radius:4px;margin-bottom:8px;font-size:13px}
.msg.error{background:rgba(248,81,73,.1);color:var(--fail);border:1px solid rgba(248,81,73,.3)}
.msg.ok{background:rgba(63,185,80,.1);color:var(--success);border:1px solid rgba(63,185,80,.3)}
</style>
</head>
<body>

<div class="header">
  <h1>Auto Claude Code Dashboard</h1>
  <div class="header-right">
    <span><span class="dot" id="conn-dot"></span><span id="conn-text">Connecting...</span></span>
    <button class="toggle-btn active" id="auto-refresh-btn" onclick="toggleAutoRefresh()">Auto-refresh</button>
    <span id="last-updated"></span>
  </div>
</div>

<div class="container">
  <!-- Status cards -->
  <div class="cards" id="status-cards"></div>

  <!-- Cycle history -->
  <h2 class="section-title">Cycle History</h2>
  <div class="filter-bar">
    <select id="filter-type" onchange="applyFilters()"><option value="">All Types</option></select>
    <select id="filter-success" onchange="applyFilters()">
      <option value="">All Results</option>
      <option value="true">Success</option>
      <option value="false">Failure</option>
    </select>
    <input type="text" id="filter-search" placeholder="Search task description..." oninput="applyFilters()">
  </div>
  <div class="table-wrap"><table>
    <thead><tr id="history-head"></tr></thead>
    <tbody id="history-body"></tbody>
  </table></div>
  <div class="pagination" id="pagination"></div>

  <!-- Feedback panel -->
  <h2 class="section-title" style="margin-top:24px">Feedback</h2>
  <div class="tabs" id="feedback-tabs"></div>
  <div id="feedback-content"></div>

  <!-- Log viewer -->
  <div class="log-section">
    <div class="log-header" onclick="toggleLog()">
      <h3>Log Viewer</h3>
      <button class="refresh-btn" onclick="event.stopPropagation();fetchLog()">Refresh</button>
    </div>
    <div class="log-panel collapsed" id="log-panel"></div>
  </div>
</div>

<script>
// State
let autoRefresh = true;
let statusData = {};
let historyData = [];
let feedbackData = {pending:[], done:[], failed:[]};
let logLines = [];
let sortCol = 'timestamp';
let sortAsc = false;
let currentPage = 0;
const pageSize = 50;
let expandedRows = new Set();
let activeTab = 'pending';
let locCache = {};
let logOpen = false;

// Polling intervals
let statusTimer, historyTimer, feedbackTimer, logTimer;

function toggleAutoRefresh(){
  autoRefresh = !autoRefresh;
  const btn = document.getElementById('auto-refresh-btn');
  btn.classList.toggle('active', autoRefresh);
  if(autoRefresh) startPolling(); else stopPolling();
}

function startPolling(){
  fetchStatus(); fetchHistory(); fetchFeedback(); if(logOpen) fetchLog();
  statusTimer = setInterval(fetchStatus, 5000);
  historyTimer = setInterval(fetchHistory, 10000);
  feedbackTimer = setInterval(fetchFeedback, 10000);
  logTimer = setInterval(()=>{ if(logOpen) fetchLog(); }, 15000);
}

function stopPolling(){
  clearInterval(statusTimer); clearInterval(historyTimer);
  clearInterval(feedbackTimer); clearInterval(logTimer);
}

// API calls
async function api(path, opts){
  try{
    const r = await fetch(path, opts);
    const d = await r.json();
    setConnected(true);
    return d;
  }catch(e){ setConnected(false); return null; }
}

function setConnected(ok){
  document.getElementById('conn-dot').className = 'dot ' + (ok?'on':'off');
  document.getElementById('conn-text').textContent = ok?'Connected':'Disconnected';
}

function updateTimestamp(){
  document.getElementById('last-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

// Status
async function fetchStatus(){
  const d = await api('/api/status');
  if(!d) return;
  statusData = d;
  renderStatus();
  updateTimestamp();
}

function renderStatus(){
  const s = statusData;
  const cards = [
    {label:'Orchestrator', value: s.running ? 'Running' : 'Stopped',
     cls: s.running ? 'success' : 'fail', bar: null},
    {label:'Consecutive Failures', value: s.consecutive_failures,
     cls: s.consecutive_failures >= (s.max_consecutive_failures||5) ? 'fail' : s.consecutive_failures > 0 ? 'warn' : 'success',
     bar: {pct: Math.min(100, (s.consecutive_failures/(s.max_consecutive_failures||5))*100), color: s.consecutive_failures >= (s.max_consecutive_failures||5) ? 'red' : 'yellow'}},
    {label:'Cycles / Hour', value: s.cycles_per_hour,
     cls: s.cycles_per_hour >= (s.max_cycles_per_hour||30) ? 'fail' : 'success',
     bar: {pct: Math.min(100, (s.cycles_per_hour/(s.max_cycles_per_hour||30))*100), color: 'blue'}},
    {label:'Cost / Hour', value: '$' + (s.cost_per_hour||0).toFixed(2),
     cls: s.cost_per_hour >= (s.max_cost_per_hour||10) ? 'fail' : 'success',
     bar: {pct: Math.min(100, (s.cost_per_hour/(s.max_cost_per_hour||10))*100), color: 'blue'}},
    {label:'Success Rate', value: (s.success_rate||0).toFixed(1) + '%',
     cls: s.success_rate >= 70 ? 'success' : s.success_rate >= 40 ? 'warn' : 'fail',
     bar: {pct: s.success_rate||0, color: s.success_rate >= 70 ? 'green' : s.success_rate >= 40 ? 'yellow' : 'red'}},
    {label:'Disk Space', value: (s.disk_free_mb||0).toFixed(0) + ' MB free',
     cls: (s.disk_free_mb||0) < (s.min_disk_mb||500) ? 'fail' : 'success',
     bar: {pct: s.disk_used_pct||0, color: (s.disk_used_pct||0) > 90 ? 'red' : 'green'}},
  ];
  const el = document.getElementById('status-cards');
  let html = cards.map(c => `
    <div class="card">
      <div class="card-label">${c.label}</div>
      <div class="card-value ${c.cls}">${c.value}</div>
      ${c.bar ? `<div class="bar-track"><div class="bar-fill ${c.bar.color}" style="width:${c.bar.pct}%"></div></div>` : ''}
    </div>`).join('');

  // Current Activity card (spans full width if active)
  const cs = s.cycle_state;
  if(cs){
    const elapsed = cs.started_at ? formatDuration((Date.now()/1000) - cs.started_at) : '-';
    const phase = cs.stale ? 'Crashed during: ' + escHtml(cs.phase) : escHtml(cs.phase);
    const phaseCls = cs.stale ? 'fail' : 'info';
    const desc = escHtml((cs.task_description||'').substring(0, 100));
    const agent = cs.pipeline_agent ? ' (' + escHtml(cs.pipeline_agent) + ')' : '';
    const cost = cs.accumulated_cost ? ' &middot; $' + cs.accumulated_cost.toFixed(4) : '';
    const retry = cs.retry_count ? ' &middot; retry ' + cs.retry_count : '';
    const batchInfo = (cs.batch_size||1) > 1 ? ' &middot; batch of ' + cs.batch_size : '';
    html += `<div class="card" style="grid-column:1/-1">
      <div class="card-label">Current Activity</div>
      <div class="card-value" style="font-size:16px;color:var(--${phaseCls})">${phase}${agent}</div>
      <div style="margin-top:6px;color:var(--muted);font-size:13px">${desc}</div>
      <div style="margin-top:4px;color:var(--muted);font-size:12px">Elapsed: ${elapsed}${cost}${retry}${batchInfo}</div>
    </div>`;
  }

  el.innerHTML = html;
}

// History
async function fetchHistory(){
  const params = new URLSearchParams({limit:'500', offset:'0'});
  const d = await api('/api/history?' + params);
  if(!d) return;
  historyData = d.records || [];
  populateTypeFilter();
  renderHistory();
  updateTimestamp();
}

function populateTypeFilter(){
  const sel = document.getElementById('filter-type');
  const current = sel.value;
  const types = [...new Set(historyData.map(r => r.task_type).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">All Types</option>' + types.map(t => `<option value="${t}">${t}</option>`).join('');
  sel.value = current;
}

function applyFilters(){
  currentPage = 0;
  expandedRows.clear();
  renderHistory();
}

function getFilteredHistory(){
  const typeF = document.getElementById('filter-type').value;
  const successF = document.getElementById('filter-success').value;
  const searchF = document.getElementById('filter-search').value.toLowerCase();
  return historyData.filter(r => {
    if(typeF && r.task_type !== typeF) return false;
    if(successF === 'true' && !r.success) return false;
    if(successF === 'false' && r.success) return false;
    if(searchF && !(r.task_description||'').toLowerCase().includes(searchF)) return false;
    return true;
  });
}

const columns = [
  {key:'success', label:'', sortable:true},
  {key:'timestamp', label:'Time', sortable:true},
  {key:'task_description', label:'Task', sortable:true},
  {key:'task_type', label:'Type', sortable:true},
  {key:'batch_size', label:'Batch', sortable:true},
  {key:'duration_seconds', label:'Duration', sortable:true},
  {key:'cost_usd', label:'Cost', sortable:true},
  {key:'commit_hash', label:'Commit', sortable:true},
  {key:'validation_summary', label:'Validation', sortable:false},
  {key:'loc', label:'LOC', sortable:false},
];

function renderHistory(){
  // Header
  const head = document.getElementById('history-head');
  head.innerHTML = columns.map(c => {
    const arrow = c.key === sortCol ? (sortAsc ? ' &#9650;' : ' &#9660;') : '';
    return `<th ${c.sortable ? `onclick="sortBy('${c.key}')"` : ''}>${c.label}<span class="sort-arrow">${arrow}</span></th>`;
  }).join('');

  let filtered = getFilteredHistory();

  // Sort
  filtered.sort((a,b) => {
    let va = a[sortCol], vb = b[sortCol];
    if(sortCol === 'success'){va = va?1:0; vb = vb?1:0;}
    if(typeof va === 'string') va = va.toLowerCase();
    if(typeof vb === 'string') vb = vb.toLowerCase();
    if(va < vb) return sortAsc ? -1 : 1;
    if(va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  // Paginate
  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  if(currentPage >= totalPages) currentPage = totalPages - 1;
  const start = currentPage * pageSize;
  const page = filtered.slice(start, start + pageSize);

  // Fetch LOC for commits on this page
  const hashes = page.map(r => r.commit_hash).filter(Boolean);
  if(hashes.length > 0){
    const missing = hashes.filter(h => !(h in locCache));
    if(missing.length > 0){
      api('/api/loc?commits=' + missing.join(',')).then(d => {
        if(d) Object.assign(locCache, d);
        updateLocCells();
      });
    }
  }

  const body = document.getElementById('history-body');
  let html = '';
  page.forEach((r, i) => {
    const idx = start + i;
    const cls = r.success ? 'success-row' : 'fail-row';
    const icon = r.success ? '<span class="status-icon" style="color:var(--success)">&#10003;</span>' : '<span class="status-icon" style="color:var(--fail)">&#10007;</span>';
    const ts = new Date(r.timestamp * 1000).toLocaleString();
    const desc = escHtml((r.task_description||'').substring(0, 80));
    const batchExtra = (r.batch_size||1) > 1 ? ` <span style="color:var(--muted);font-size:11px">(+ ${r.batch_size - 1} more)</span>` : '';
    const typeBadge = r.task_type ? `<span class="badge badge-${r.task_type}">${r.task_type}</span>` : '';
    const dur = r.duration_seconds ? formatDuration(r.duration_seconds) : '-';
    const cost = r.cost_usd ? '$' + r.cost_usd.toFixed(4) : '-';
    const commit = r.commit_hash ? r.commit_hash.substring(0,7) : '-';
    const validation = escHtml((r.validation_summary||'').substring(0, 40));
    const loc = r.commit_hash && locCache[r.commit_hash] ? formatLoc(locCache[r.commit_hash]) : '-';

    html += `<tr class="${cls} expandable" onclick="toggleRow(${idx})">
      <td>${icon}</td><td>${ts}</td><td>${desc}${batchExtra}</td><td>${typeBadge}</td>
      <td>${r.batch_size||1}</td><td>${dur}</td><td>${cost}</td><td><code>${commit}</code></td>
      <td>${validation}</td><td class="loc-cell" data-hash="${r.commit_hash||''}">${loc}</td>
    </tr>`;
    if(expandedRows.has(idx)){
      html += renderDetailRow(r);
    }
  });
  body.innerHTML = html || '<tr><td colspan="10"><div class="empty-state">No cycles recorded yet.</div></td></tr>';

  // Pagination
  const pag = document.getElementById('pagination');
  if(totalPages > 1){
    pag.innerHTML = `<button onclick="changePage(-1)" ${currentPage===0?'disabled':''}>Prev</button>
      <span>Page ${currentPage+1} of ${totalPages} (${total} records)</span>
      <button onclick="changePage(1)" ${currentPage>=totalPages-1?'disabled':''}>Next</button>`;
  } else {
    pag.innerHTML = `<span>${total} records</span>`;
  }
}

function renderDetailRow(r){
  let details = '';

  // Batch task list with type badges and source files
  if(r.task_descriptions && r.task_descriptions.length > 1){
    details += '<strong>Tasks:</strong>';
    r.task_descriptions.forEach((d, i) => {
      const ttype = (r.task_types && r.task_types[i]) || '';
      const badge = ttype ? ` <span class="badge badge-${ttype}">${ttype}</span>` : '';
      const srcFile = (r.task_source_files && r.task_source_files[i]) ? r.task_source_files[i] : '';
      const srcLine = (r.task_line_numbers && r.task_line_numbers[i] != null) ? ':' + r.task_line_numbers[i] : '';
      const srcRef = srcFile ? ` <span style="color:var(--muted);font-size:11px">(${escHtml(srcFile + srcLine)})</span>` : '';
      details += `\n  ${i+1}. ${escHtml(d)}${badge}${srcRef}`;
    });
  } else {
    details += `<strong>Task:</strong>\n${escHtml(r.task_description||'')}`;
    if(r.task_source_files && r.task_source_files[0]){
      const srcLine = (r.task_line_numbers && r.task_line_numbers[0] != null) ? ':' + r.task_line_numbers[0] : '';
      details += `\n<strong>Source:</strong> ${escHtml(r.task_source_files[0] + srcLine)}`;
    }
  }

  // Cost and duration
  if(r.cost_usd || r.duration_seconds){
    details += '\n';
    if(r.cost_usd) details += `\n<strong>Cost:</strong> $${r.cost_usd.toFixed(4)}`;
    if(r.duration_seconds) details += `  <strong>Duration:</strong> ${formatDuration(r.duration_seconds)}`;
  }

  if(r.error) details += `\n\n<strong>Error:</strong>\n${escHtml(r.error)}`;
  if(r.pipeline_mode) details += `\n\n<strong>Pipeline:</strong> mode=${escHtml(r.pipeline_mode)}, revisions=${r.pipeline_revision_count||0}, approved=${r.pipeline_review_approved}`;
  if(r.validation_summary) details += `\n\n<strong>Validation:</strong> ${escHtml(r.validation_summary)}`;
  if(r.validation_retry_count) details += `\n<strong>Validation retries:</strong> ${r.validation_retry_count}`;
  if(r.commit_hash && locCache[r.commit_hash]){
    const loc = locCache[r.commit_hash];
    if(loc.files && loc.files.length){
      details += '\n\n<strong>Files changed:</strong>';
      loc.files.forEach(f => { details += `\n  ${f.path}: +${f.insertions} -${f.deletions}`; });
    }
  }
  return `<tr class="detail-row"><td colspan="10"><pre>${details}</pre></td></tr>`;
}

function updateLocCells(){
  document.querySelectorAll('.loc-cell').forEach(cell => {
    const hash = cell.dataset.hash;
    if(hash && locCache[hash]) cell.textContent = formatLoc(locCache[hash]);
  });
}

function formatLoc(loc){
  if(!loc) return '-';
  return `+${loc.total_insertions||0} -${loc.total_deletions||0}`;
}

function formatDuration(sec){
  if(sec < 60) return sec.toFixed(0) + 's';
  if(sec < 3600) return (sec/60).toFixed(1) + 'm';
  return (sec/3600).toFixed(1) + 'h';
}

function escHtml(s){
  const d=document.createElement('div'); d.textContent=s; return d.innerHTML;
}

function sortBy(col){
  if(sortCol===col) sortAsc=!sortAsc; else {sortCol=col; sortAsc=col==='timestamp'?false:true;}
  renderHistory();
}

function toggleRow(idx){
  if(expandedRows.has(idx)) expandedRows.delete(idx); else expandedRows.add(idx);
  renderHistory();
}

function changePage(delta){
  currentPage += delta;
  expandedRows.clear();
  renderHistory();
}

// Feedback
async function fetchFeedback(){
  const d = await api('/api/feedback');
  if(!d) return;
  feedbackData = d;
  renderFeedback();
  updateTimestamp();
}

function renderFeedback(){
  const tabs = document.getElementById('feedback-tabs');
  const counts = {pending: feedbackData.pending?.length||0, done: feedbackData.done?.length||0, failed: feedbackData.failed?.length||0};
  tabs.innerHTML = ['pending','done','failed'].map(t =>
    `<div class="tab ${activeTab===t?'active':''}" onclick="switchTab('${t}')">${t.charAt(0).toUpperCase()+t.slice(1)}<span class="tab-count">${counts[t]}</span></div>`
  ).join('');

  const content = document.getElementById('feedback-content');
  let html = '';

  if(activeTab === 'pending'){
    html += `<div class="feedback-form">
      <input type="text" id="fb-filename" placeholder="filename.md" maxlength="100">
      <textarea id="fb-content" placeholder="Describe the task for Claude..."></textarea>
      <button onclick="submitFeedback()">Submit Feedback</button>
      <div id="fb-msg"></div>
    </div>`;
  }

  const items = feedbackData[activeTab] || [];
  if(items.length === 0){
    html += '<div class="empty-state">No ' + activeTab + ' feedback items.</div>';
  } else {
    items.forEach(item => {
      const preview = (item.content||'').substring(0, 200);
      const full = item.content||'';
      const hasMore = full.length > 200;
      html += `<div class="feedback-item">
        <div class="feedback-item-header">
          <span class="feedback-item-name">${escHtml(item.name)}</span>
          <span>
            ${activeTab==='pending' ? `<button class="delete-btn" onclick="deleteFeedback('${escHtml(item.name)}')">Delete</button>` : ''}
          </span>
        </div>
        <div class="feedback-item-content" id="fb-${escHtml(item.name)}">${escHtml(preview)}${hasMore ? '...' : ''}</div>
        ${hasMore ? `<button class="expand-btn" onclick="toggleFbExpand(this, '${escHtml(item.name)}')">Show more</button>` : ''}
      </div>`;
    });
  }
  content.innerHTML = html;
}

function switchTab(t){ activeTab=t; renderFeedback(); }

let fbExpanded = {};
function toggleFbExpand(btn, name){
  const el = document.getElementById('fb-' + name);
  if(!el) return;
  const item = (feedbackData[activeTab]||[]).find(i=>i.name===name);
  if(!item) return;
  if(fbExpanded[name]){
    el.textContent = item.content.substring(0,200) + (item.content.length>200?'...':'');
    btn.textContent = 'Show more';
    fbExpanded[name] = false;
  } else {
    el.textContent = item.content;
    btn.textContent = 'Show less';
    fbExpanded[name] = true;
  }
}

async function submitFeedback(){
  const nameEl = document.getElementById('fb-filename');
  const contentEl = document.getElementById('fb-content');
  const msgEl = document.getElementById('fb-msg');
  const name = nameEl.value.trim();
  const content = contentEl.value.trim();
  if(!name || !content){ msgEl.innerHTML='<div class="msg error">Filename and content are required.</div>'; return; }
  const r = await fetch('/api/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:name, content:content})});
  const d = await r.json();
  if(d.error){ msgEl.innerHTML=`<div class="msg error">${escHtml(d.error)}</div>`; }
  else { msgEl.innerHTML='<div class="msg ok">Feedback submitted.</div>'; nameEl.value=''; contentEl.value=''; fetchFeedback(); }
}

async function deleteFeedback(name){
  if(!confirm('Delete feedback file "'+name+'"?')) return;
  await fetch('/api/feedback/'+encodeURIComponent(name), {method:'DELETE'});
  fetchFeedback();
}

// Log
async function fetchLog(){
  const d = await api('/api/log?lines=100');
  if(!d) return;
  logLines = d.lines || [];
  renderLog();
}

function renderLog(){
  const panel = document.getElementById('log-panel');
  panel.innerHTML = logLines.map(l => {
    let cls = 'info';
    if(l.includes('[ERROR]')) cls = 'error';
    else if(l.includes('[WARNING]')) cls = 'warning';
    return `<div class="log-line ${cls}">${escHtml(l)}</div>`;
  }).join('');
  panel.scrollTop = panel.scrollHeight;
}

function toggleLog(){
  logOpen = !logOpen;
  document.getElementById('log-panel').classList.toggle('collapsed', !logOpen);
  if(logOpen) fetchLog();
}

// Init
startPolling();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section 2: Data access functions
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    """Load config YAML and return raw dict + parsed values needed by dashboard."""
    import yaml  # noqa: delay import — only dependency
    result: Dict[str, Any] = {
        "target_dir": ".",
        "history_file": "state/history.json",
        "state_dir": "state",
        "lock_file": "state/lock.pid",
        "log_file": "state/auto_claude.log",
        "feedback_dir": "feedback",
        "feedback_done_dir": "feedback/done",
        "feedback_failed_dir": "feedback/failed",
        "max_consecutive_failures": 5,
        "max_cycles_per_hour": 30,
        "max_cost_usd_per_hour": 10.0,
        "min_disk_space_mb": 500,
    }
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if not raw or not isinstance(raw, dict):
            return result
        result["target_dir"] = raw.get("target_dir", ".")
        paths = raw.get("paths", {})
        if isinstance(paths, dict):
            result["history_file"] = paths.get("history_file", result["history_file"])
            result["lock_file"] = paths.get("lock_file", result["lock_file"])
            result["feedback_dir"] = paths.get("feedback_dir", result["feedback_dir"])
            result["feedback_done_dir"] = paths.get("feedback_done_dir", result["feedback_done_dir"])
            result["feedback_failed_dir"] = paths.get("feedback_failed_dir", result["feedback_failed_dir"])
            # Derive state_dir from history_file parent
            result["state_dir"] = str(Path(result["history_file"]).parent)
        logging_cfg = raw.get("logging", {})
        if isinstance(logging_cfg, dict):
            result["log_file"] = logging_cfg.get("file", result["log_file"])
        safety = raw.get("safety", {})
        if isinstance(safety, dict):
            result["max_consecutive_failures"] = safety.get("max_consecutive_failures", result["max_consecutive_failures"])
            result["max_cycles_per_hour"] = safety.get("max_cycles_per_hour", result["max_cycles_per_hour"])
            result["max_cost_usd_per_hour"] = safety.get("max_cost_usd_per_hour", result["max_cost_usd_per_hour"])
            result["min_disk_space_mb"] = safety.get("min_disk_space_mb", result["min_disk_space_mb"])
    except Exception as e:
        logger.warning("Failed to load config %s: %s", config_path, e)
    return result


def load_history(history_path: str) -> List[Dict[str, Any]]:
    """Load history.json directly (read-only, no StateManager side-effects)."""
    p = Path(history_path)
    if not p.exists():
        return []
    try:
        text = p.read_text().strip()
        if not text:
            return []
        records = json.loads(text)
        if isinstance(records, list):
            return records
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _read_cycle_state(state_dir: str) -> Optional[Dict[str, Any]]:
    """Read current_cycle.json from state_dir. Returns None if no active cycle."""
    p = Path(state_dir) / "current_cycle.json"
    if not p.exists():
        return None
    try:
        text = p.read_text().strip()
        if not text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return None


def is_orchestrator_running(lock_path: str) -> Tuple[bool, Optional[int]]:
    """Check if orchestrator PID in lock file is alive. Read-only — no flock."""
    p = Path(lock_path)
    if not p.exists():
        return False, None
    try:
        text = p.read_text().strip()
        if not text:
            return False, None
        pid = int(text)
        os.kill(pid, 0)  # signal 0 = check existence
        return True, pid
    except (ValueError, ProcessLookupError, OSError):
        return False, None


def _get_cycle_state_for_api(cfg: Dict[str, Any], running: bool) -> Optional[Dict[str, Any]]:
    """Get cycle state for the API, handling stale state from crashed orchestrator."""
    state = _read_cycle_state(cfg["state_dir"])
    if state is None:
        return None
    # If orchestrator is not running but state file exists, it crashed mid-cycle
    if not running:
        state["stale"] = True
    return state


def compute_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute aggregate status data for the /api/status endpoint."""
    records = load_history(cfg["history_file"])
    running, pid = is_orchestrator_running(cfg["lock_file"])
    now = time.time()
    cutoff = now - 3600

    # Consecutive failures (reverse scan)
    consecutive_failures = 0
    for r in reversed(records):
        if r.get("success", False):
            break
        consecutive_failures += 1

    # Cycles/hour
    cycles_per_hour = sum(1 for r in records if r.get("timestamp", 0) >= cutoff)

    # Cost/hour
    cost_per_hour = sum(r.get("cost_usd", 0.0) for r in records if r.get("timestamp", 0) >= cutoff)

    # Success rate
    total = len(records)
    successes = sum(1 for r in records if r.get("success", False))
    success_rate = (successes / total * 100) if total > 0 else 0.0

    # Disk space
    try:
        usage = shutil.disk_usage(cfg["target_dir"])
        disk_free_mb = usage.free / (1024 * 1024)
        disk_used_pct = (usage.used / usage.total * 100) if usage.total > 0 else 0
    except OSError:
        disk_free_mb = 0
        disk_used_pct = 0

    return {
        "running": running,
        "pid": pid,
        "consecutive_failures": consecutive_failures,
        "max_consecutive_failures": cfg["max_consecutive_failures"],
        "cycles_per_hour": cycles_per_hour,
        "max_cycles_per_hour": cfg["max_cycles_per_hour"],
        "cost_per_hour": round(cost_per_hour, 4),
        "max_cost_per_hour": cfg["max_cost_usd_per_hour"],
        "success_rate": round(success_rate, 1),
        "total_cycles": total,
        "total_successes": successes,
        "disk_free_mb": round(disk_free_mb, 0),
        "disk_used_pct": round(disk_used_pct, 1),
        "min_disk_mb": cfg["min_disk_space_mb"],
        "cycle_state": _get_cycle_state_for_api(cfg, running),
    }


def get_feedback_files(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """List feedback files from pending/done/failed directories."""
    result: Dict[str, List[Dict[str, str]]] = {"pending": [], "done": [], "failed": []}

    dirs = {
        "pending": cfg["feedback_dir"],
        "done": cfg["feedback_done_dir"],
        "failed": cfg["feedback_failed_dir"],
    }

    for category, dirpath in dirs.items():
        p = Path(dirpath)
        if not p.is_dir():
            continue
        for f in sorted(p.iterdir()):
            if not f.is_file() or f.suffix not in (".md", ".txt") or f.name == ".gitkeep":
                continue
            try:
                content = f.read_text()
            except OSError:
                content = "(unreadable)"
            result[category].append({"name": f.name, "content": content})

    return result


def get_loc_for_commits(
    target_dir: str,
    hashes: List[str],
    cache: Dict[str, Any],
    lock: threading.Lock,
) -> Dict[str, Any]:
    """Get lines-of-code changed per commit hash via git log --numstat."""
    result: Dict[str, Any] = {}
    to_fetch: List[str] = []

    with lock:
        for h in hashes[:MAX_LOC_HASHES_PER_REQUEST]:
            if h in cache:
                result[h] = cache[h]
            else:
                to_fetch.append(h)

    for h in to_fetch:
        # Validate hash is hex only (safety)
        if not re.match(r'^[0-9a-fA-F]{4,40}$', h):
            result[h] = {"error": "invalid hash"}
            continue
        try:
            proc = subprocess.run(
                ["git", "log", "--numstat", "--format=", "-1", h],
                cwd=target_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            files = []
            total_ins = 0
            total_del = 0
            for line in proc.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) == 3:
                    ins_str, del_str, path = parts
                    ins = int(ins_str) if ins_str != "-" else 0
                    dels = int(del_str) if del_str != "-" else 0
                    files.append({"path": path, "insertions": ins, "deletions": dels})
                    total_ins += ins
                    total_del += dels
            loc_data = {
                "total_insertions": total_ins,
                "total_deletions": total_del,
                "files": files,
            }
            result[h] = loc_data
            with lock:
                cache[h] = loc_data
        except (subprocess.TimeoutExpired, OSError):
            result[h] = {"error": "git failed"}

    return result


def read_log_tail(log_path: str, num_lines: int) -> List[str]:
    """Read the last N lines from the log file."""
    num_lines = min(num_lines, MAX_LOG_LINES)
    p = Path(log_path)
    if not p.exists():
        return []
    try:
        text = p.read_text()
        lines = text.splitlines()
        return lines[-num_lines:]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Section 3: HTTP handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    # Shared state (set by server setup)
    dashboard_cfg: Dict[str, Any] = {}
    loc_cache: Dict[str, Any] = {}
    loc_lock: threading.Lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging to stderr."""
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        routes = {
            "/": self._serve_spa,
            "/api/status": self._api_status,
            "/api/history": self._api_history,
            "/api/loc": self._api_loc,
            "/api/feedback": self._api_feedback_list,
            "/api/log": self._api_log,
        }

        handler = routes.get(path)
        if handler:
            handler(query)
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/feedback":
            self._api_feedback_submit()
        else:
            self._send_error(404, "Not found")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path.startswith("/api/feedback/"):
            name = path[len("/api/feedback/"):]
            self._api_feedback_delete(name)
        else:
            self._send_error(404, "Not found")

    # ---- Route handlers ----

    def _serve_spa(self, query: Dict) -> None:
        content = STATIC_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _api_status(self, query: Dict) -> None:
        status = compute_status(self.dashboard_cfg)
        self._send_json(status)

    def _api_history(self, query: Dict) -> None:
        limit = min(int(query.get("limit", ["500"])[0]), 5000)
        offset = int(query.get("offset", ["0"])[0])

        records = load_history(self.dashboard_cfg["history_file"])
        # Return newest first by default
        records = list(reversed(records))

        # Server-side filtering
        success_filter = query.get("success", [""])[0]
        type_filter = query.get("type", [""])[0]
        search_filter = query.get("search", [""])[0].lower()

        if success_filter:
            success_val = success_filter.lower() == "true"
            records = [r for r in records if r.get("success", False) == success_val]
        if type_filter:
            records = [r for r in records if r.get("task_type") == type_filter]
        if search_filter:
            records = [r for r in records if search_filter in (r.get("task_description", "") or "").lower()]

        total = len(records)
        page = records[offset:offset + limit]

        self._send_json({"total": total, "offset": offset, "limit": limit, "records": page})

    def _api_loc(self, query: Dict) -> None:
        commits_str = query.get("commits", [""])[0]
        if not commits_str:
            self._send_json({})
            return
        hashes = [h.strip() for h in commits_str.split(",") if h.strip()]
        result = get_loc_for_commits(
            self.dashboard_cfg["target_dir"],
            hashes,
            self.loc_cache,
            self.loc_lock,
        )
        self._send_json(result)

    def _api_feedback_list(self, query: Dict) -> None:
        fb = get_feedback_files(self.dashboard_cfg)
        self._send_json(fb)

    def _api_feedback_submit(self) -> None:
        body = self._read_body()
        if body is None:
            return

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        filename = data.get("filename", "").strip()
        content = data.get("content", "").strip()

        if not filename or not content:
            self._send_error(400, "filename and content are required")
            return

        # Validate filename
        if not FEEDBACK_FILENAME_RE.match(filename):
            self._send_error(400, "Invalid filename. Use alphanumeric/hyphens/underscores/dots, ending with .md or .txt, max 100 chars.")
            return

        if ".." in filename or "/" in filename:
            self._send_error(400, "Invalid filename: path traversal not allowed")
            return

        if len(content) > MAX_FEEDBACK_CONTENT_SIZE:
            self._send_error(400, f"Content too large (max {MAX_FEEDBACK_CONTENT_SIZE // 1024}KB)")
            return

        feedback_dir = Path(self.dashboard_cfg["feedback_dir"])
        feedback_dir.mkdir(parents=True, exist_ok=True)
        target = feedback_dir / filename

        if target.exists():
            self._send_error(409, f"File '{filename}' already exists")
            return

        try:
            target.write_text(content)
        except OSError as e:
            self._send_error(500, f"Failed to write file: {e}")
            return

        self._send_json({"ok": True, "filename": filename})

    def _api_feedback_delete(self, name: str) -> None:
        if not name or ".." in name or "/" in name:
            self._send_error(400, "Invalid filename")
            return

        feedback_dir = Path(self.dashboard_cfg["feedback_dir"])
        target = feedback_dir / name

        if not target.exists():
            self._send_error(404, f"File '{name}' not found")
            return

        # Only allow deleting from the pending directory
        try:
            target.resolve().relative_to(feedback_dir.resolve())
        except ValueError:
            self._send_error(403, "Cannot delete files outside feedback directory")
            return

        try:
            target.unlink()
        except OSError as e:
            self._send_error(500, f"Failed to delete: {e}")
            return

        self._send_json({"ok": True, "deleted": name})

    def _api_log(self, query: Dict) -> None:
        num_lines = min(int(query.get("lines", ["100"])[0]), MAX_LOG_LINES)
        lines = read_log_tail(self.dashboard_cfg["log_file"], num_lines)
        self._send_json({"lines": lines})

    # ---- Helpers ----

    def _send_json(self, data: Any) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, code: int, message: str) -> None:
        content = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _read_body(self) -> Optional[str]:
        length_str = self.headers.get("Content-Length", "0")
        try:
            length = int(length_str)
        except ValueError:
            self._send_error(400, "Invalid Content-Length")
            return None
        if length > MAX_FEEDBACK_CONTENT_SIZE + 1024:  # extra for JSON overhead
            self._send_error(413, "Request body too large")
            return None
        try:
            return self.rfile.read(length).decode("utf-8")
        except (OSError, UnicodeDecodeError) as e:
            self._send_error(400, f"Failed to read body: {e}")
            return None

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ---------------------------------------------------------------------------
# Section 4: Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto Claude Code Web Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_config(args.config)
    DashboardHandler.dashboard_cfg = cfg
    DashboardHandler.loc_cache = {}
    DashboardHandler.loc_lock = threading.Lock()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)
    logger.info("Dashboard running at http://localhost:%d", args.port)

    # Graceful shutdown on SIGINT / SIGTERM
    def shutdown_handler(signum: int, frame: Any) -> None:
        logger.info("Shutting down dashboard...")
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        logger.info("Dashboard stopped.")


if __name__ == "__main__":
    main()
