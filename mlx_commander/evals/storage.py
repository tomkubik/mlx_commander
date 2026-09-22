"""
3-Tier Storage Architecture for Generative Evaluation Results.
Supports:
  - Tier 1: Master eval_leaderboard.csv at adapters root (easy to open in Numbers/Excel).
  - Tier 2: Per-run eval_summary.json & eval_predictions.jsonl in run folder.
  - Tier 3: Zero-dependency interactive eval_comparison.html dashboard for Safari/Chrome.
"""

import csv
import html
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def append_to_leaderboard_csv(
    csv_path: Path,
    run_dict: Dict[str, Any],
    metrics_dict: Dict[str, Any],
) -> None:
    """Append a summary row to the global eval_leaderboard.csv."""
    fieldnames = [
        "run_name",
        "model",
        "fine_tune_type",
        "iters",
        "batch_size",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "exact_match_pct",
        "substring_match_pct",
        "avg_word_f1",
        "avg_word_prec",
        "avg_word_recall",
        "fixed_count",
        "regressed_count",
        "total_samples",
        "tokens_per_sec",
        "timestamp",
        "adapter_path",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    row = {
        "run_name": run_dict.get("name", ""),
        "model": run_dict.get("model", ""),
        "fine_tune_type": run_dict.get("fine_tune_type", "lora"),
        "iters": run_dict.get("iters", 0),
        "batch_size": run_dict.get("batch_size", 0),
        "learning_rate": run_dict.get("learning_rate", 0.0),
        "lora_rank": run_dict.get("lora_rank", 0),
        "lora_alpha": run_dict.get("lora_alpha", 0.0),
        "exact_match_pct": metrics_dict.get("exact_match_pct", 0.0),
        "substring_match_pct": metrics_dict.get("substring_match_pct", 0.0),
        "avg_word_f1": metrics_dict.get("avg_word_f1", 0.0),
        "avg_word_prec": metrics_dict.get("avg_word_prec", 0.0),
        "avg_word_recall": metrics_dict.get("avg_word_recall", 0.0),
        "fixed_count": metrics_dict.get("fixed_count", 0),
        "regressed_count": metrics_dict.get("regressed_count", 0),
        "total_samples": metrics_dict.get("total_samples", 0),
        "tokens_per_sec": metrics_dict.get("tokens_per_sec", 0.0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapter_path": run_dict.get("adapter_path", ""),
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_run_predictions(
    run_dir: Path,
    summary_data: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    """Save per-run eval_summary.json and eval_predictions.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = run_dir / "eval_summary.json"
    pred_file = run_dir / "eval_predictions.jsonl"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(pred_file, "w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary_file, pred_file


def generate_html_dashboard(
    dashboard_path: Path,
    leaderboard_csv_path: Optional[Path] = None,
    current_run_summary: Optional[Dict[str, Any]] = None,
    current_predictions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Generate standalone offline HTML dashboard for inspecting evaluation results."""
    leaderboard_rows: List[Dict[str, str]] = []
    if leaderboard_csv_path and leaderboard_csv_path.exists():
        try:
            with open(leaderboard_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                leaderboard_rows = list(reader)
        except Exception:
            pass

    current_summary_json = json.dumps(current_run_summary or {})
    current_preds_json = json.dumps(current_predictions or [])
    leaderboard_json = json.dumps(leaderboard_rows)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLX Commander :: Generative Eval Dashboard</title>
<style>
  :root {{
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-bright: #ffffff;
    --accent: #58a6ff;
    --success: #3fb950;
    --danger: #f85149;
    --warning: #d29922;
    --dim: #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  h1 {{ font-size: 24px; color: var(--text-bright); }}
  .tag {{
    background: #1f6feb22;
    color: var(--accent);
    border: 1px solid #388bfd66;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 12px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
  }}
  .grid-4 {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }}
  .metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }}
  .metric-val {{
    font-size: 28px;
    font-weight: bold;
    color: var(--text-bright);
    margin-top: 4px;
  }}
  .metric-lbl {{ font-size: 13px; color: var(--dim); }}
  .status-preserved {{ color: var(--success); font-weight: bold; }}
  .status-fixed {{ color: var(--accent); font-weight: bold; }}
  .status-regressed {{ color: var(--danger); font-weight: bold; }}
  .status-persistent {{ color: var(--warning); }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th, td {{
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: #21262d;
    color: var(--text-bright);
  }}
  tr:hover {{ background: #1f242c; }}
  .filter-bar {{
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
  }}
  .btn {{
    background: #21262d;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
  }}
  .btn.active, .btn:hover {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }}
  .sample-box {{
    background: #090d13;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
    font-size: 13px;
  }}
  .sample-row {{ margin-bottom: 6px; }}
  .sample-lbl {{ font-weight: bold; color: var(--dim); display: inline-block; width: 120px; }}
  .code-text {{ font-family: monospace; white-space: pre-wrap; word-break: break-word; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>MLX Commander :: Generative Eval Dashboard</h1>
    <div style="font-size: 13px; color: var(--dim); margin-top: 4px;">Deterministic Model Outputs & Error Analysis</div>
  </div>
  <div><span class="tag">Apple Silicon MLX</span></div>
</header>

<div class="grid-4" id="statsGrid"></div>

<div class="card">
  <h3 style="margin-bottom: 12px; color: var(--text-bright);">Evaluation Leaderboard (All Runs)</h3>
  <div style="overflow-x: auto;">
    <table id="leaderboardTable">
      <thead>
        <tr>
          <th>Run Name</th>
          <th>Model</th>
          <th>Rank</th>
          <th>LR</th>
          <th>Batch</th>
          <th>Iters</th>
          <th>Exact Match</th>
          <th>Substring Match</th>
          <th>Word F1</th>
          <th>Fixed</th>
          <th>Regressed</th>
          <th>Speed</th>
          <th>Timestamp</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="card">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
    <h3 style="color: var(--text-bright);">Test Set Predictions & Regression Explorer</h3>
    <div class="filter-bar" id="filterBar">
      <button class="btn active" onclick="setFilter('ALL')">All Samples</button>
      <button class="btn" onclick="setFilter('REGRESSED')">⚠️ Regressed (Model Broke)</button>
      <button class="btn" onclick="setFilter('FIXED')">❇️ Fixed (LoRA Healed)</button>
      <button class="btn" onclick="setFilter('PRESERVED')">Preserved</button>
      <button class="btn" onclick="setFilter('PERSISTENT_FAIL')">Persistent Fail</button>
    </div>
  </div>
  <div id="samplesList"></div>
</div>

<script>
  const currentSummary = {current_summary_json};
  const currentPredictions = {current_preds_json};
  const leaderboard = {leaderboard_json};

  let activeFilter = 'ALL';

  function initStats() {{
    const grid = document.getElementById('statsGrid');
    const em = currentSummary.exact_match_pct !== undefined ? currentSummary.exact_match_pct + '%' : 'N/A';
    const f1 = currentSummary.avg_word_f1 !== undefined ? (currentSummary.avg_word_f1 * 100).toFixed(1) + '%' : 'N/A';
    const fixed = currentSummary.fixed_count !== undefined ? currentSummary.fixed_count : '0';
    const regressed = currentSummary.regressed_count !== undefined ? currentSummary.regressed_count : '0';
    const tps = currentSummary.tokens_per_sec !== undefined ? currentSummary.tokens_per_sec.toFixed(1) + ' t/s' : 'N/A';

    grid.innerHTML = `
      <div class="metric-card"><div class="metric-lbl">Exact Match</div><div class="metric-val" style="color: var(--accent);">${{em}}</div></div>
      <div class="metric-card"><div class="metric-lbl">Word-Level F1 Score</div><div class="metric-val" style="color: var(--success);">${{f1}}</div></div>
      <div class="metric-card"><div class="metric-lbl">Cured / Fixed Prompts</div><div class="metric-val" style="color: var(--accent);">${{fixed}}</div></div>
      <div class="metric-card"><div class="metric-lbl">Regressed Prompts</div><div class="metric-val" style="color: var(--danger);">${{regressed}}</div></div>
    `;
  }}

  function initLeaderboard() {{
    const tbody = document.querySelector('#leaderboardTable tbody');
    if (!leaderboard.length) {{
      tbody.innerHTML = '<tr><td colspan="13" style="text-align:center; color: var(--dim);">No runs recorded in leaderboard yet.</td></tr>';
      return;
    }}
    tbody.innerHTML = leaderboard.map(r => `
      <tr>
        <td><strong>${{r.run_name || ''}}</strong></td>
        <td>${{r.model || ''}}</td>
        <td>${{r.lora_rank || ''}}</td>
        <td>${{r.learning_rate || ''}}</td>
        <td>${{r.batch_size || ''}}</td>
        <td>${{r.iters || ''}}</td>
        <td><strong>${{r.exact_match_pct || '0'}}%</strong></td>
        <td>${{r.substring_match_pct || '0'}}%</td>
        <td style="color: var(--success); font-weight: bold;">${{r.avg_word_f1 || '0'}}</td>
        <td style="color: var(--accent); font-weight: bold;">${{r.fixed_count || '0'}}</td>
        <td style="color: var(--danger); font-weight: bold;">${{r.regressed_count || '0'}}</td>
        <td>${{r.tokens_per_sec || '0'}} t/s</td>
        <td style="color: var(--dim);">${{r.timestamp || ''}}</td>
      </tr>
    `).join('');
  }}

  function setFilter(filter) {{
    activeFilter = filter;
    document.querySelectorAll('#filterBar .btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    renderSamples();
  }}

  function renderSamples() {{
    const container = document.getElementById('samplesList');
    const filtered = currentPredictions.filter(s => activeFilter === 'ALL' || s.status === activeFilter);
    if (!filtered.length) {{
      container.innerHTML = `<div style="text-align:center; padding: 24px; color: var(--dim);">No samples match filter '${{activeFilter}}'.</div>`;
      return;
    }}

    container.innerHTML = filtered.slice(0, 100).map(s => {{
      let statusClass = 'status-preserved';
      if (s.status === 'FIXED') statusClass = 'status-fixed';
      else if (s.status === 'REGRESSED') statusClass = 'status-regressed';
      else if (s.status === 'PERSISTENT_FAIL') statusClass = 'status-persistent';

      return `
        <div class="sample-box">
          <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <div><strong>Sample #${{s.id}}</strong> &bull; Word F1: <strong style="color:var(--success);">${{s.word_f1}}</strong></div>
            <div class="${{statusClass}}">${{s.status}}</div>
          </div>
          <div class="sample-row"><span class="sample-lbl">Prompt:</span><span class="code-text" style="color: var(--accent);">${{escapeHtml(s.prompt)}}</span></div>
          <div class="sample-row"><span class="sample-lbl">Golden:</span><span class="code-text" style="color: var(--success);">${{escapeHtml(s.golden)}}</span></div>
          ${{s.baseline_output ? `<div class="sample-row"><span class="sample-lbl">Baseline:</span><span class="code-text" style="color: var(--dim);">${{escapeHtml(s.baseline_output)}}</span></div>` : ''}}
          <div class="sample-row"><span class="sample-lbl">Generated:</span><span class="code-text" style="color: var(--text-bright);">${{escapeHtml(s.model_output)}}</span></div>
        </div>
      `;
    }}).join('');
  }}

  function escapeHtml(text) {{
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }}

  initStats();
  initLeaderboard();
  renderSamples();
</script>
</body>
</html>
"""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)
