#!/usr/bin/env python3
"""Bio-Cascade SLM V15 & OpenAI GPT-5-Nano 100 English Wikinews Benchmark Suite.

Features:
- Models: Bio-Cascade SLM V15 (models/bio_cascade_v15_1_milestone.pkl)
- Questions: Predefined General Domain (n=20) + Complex Reasoning (gpt-5-nano)
- Strict binary evaluation ("Answer strictly Yes or No.")
- Independent judging via OpenAI gpt-5-nano
- Live real-time Web Dashboard with Chart.js on http://127.0.0.1:7865/
- Checkpoint evaluation at Article 50 & Final at Article 100.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder, clean_news_context

TOKENIZER_PATH = BASE_DIR / "models" / "bio_bpe_32k.json"
V15_MILESTONE = BASE_DIR / "models" / "bio_cascade_v15_1_milestone.pkl"
CACHE_FILE = BASE_DIR / "logs" / "benchmark_200_news_cache.json"
STATE_FILE = BASE_DIR / "logs" / "live_v15_1_benchmark.json"
LIVE_COMPAT_FILE = BASE_DIR / "logs" / "live_interrogation_200_gpt5.json"
PORT = 7865

# Load OpenAI API Key from .env or environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                OPENAI_API_KEY = line.split("=", 1)[1].strip()
                break

# ==============================================================================
# 20 PREDEFINED GENERAL DOMAIN YES/NO QUESTIONS (n=20)
# ==============================================================================
GENERAL_20_QUESTIONS = [
    {
        "id": "financial_crime",
        "category": "Financial Crime & Fraud",
        "question": "Is this article about financial crime, corporate fraud, embezzlement, or bribery? Answer strictly Yes or No.",
    },
    {
        "id": "education",
        "category": "Education & Academia",
        "question": "Is this article about primary, secondary, or higher education, schools, or universities? Answer strictly Yes or No.",
    },
    {
        "id": "personal_development",
        "category": "Personal Development & Coaching",
        "question": "Is this article about personal development, self-improvement, coaching, or psychological well-being? Answer strictly Yes or No.",
    },
    {
        "id": "armed_conflict",
        "category": "Military & Armed Conflict",
        "question": "Is this article about armed military warfare, battlefield combat, or armed insurgency? Answer strictly Yes or No.",
    },
    {
        "id": "healthcare_medicine",
        "category": "Medicine & Healthcare",
        "question": "Is this article about medical healthcare, disease pathology, clinical surgery, or pharmaceutical treatments? Answer strictly Yes or No.",
    },
    {
        "id": "space_astronomy",
        "category": "Space Exploration & Astronomy",
        "question": "Is this article about space exploration, rocket launches, or astronomical telescope discoveries? Answer strictly Yes or No.",
    },
    {
        "id": "environmental_climate",
        "category": "Environment & Climate",
        "question": "Is this article about environmental ecology, climate change, wildlife conservation, or natural habitats? Answer strictly Yes or No.",
    },
    {
        "id": "artificial_intelligence",
        "category": "Artificial Intelligence & Robotics",
        "question": "Is this article about artificial intelligence, machine learning models, autonomous robotics, or algorithmic automation? Answer strictly Yes or No.",
    },
    {
        "id": "criminal_justice",
        "category": "Crime & Law Enforcement",
        "question": "Is this article about criminal arrests, police investigations, courtroom trials, or judicial sentencing? Answer strictly Yes or No.",
    },
    {
        "id": "macroeconomics",
        "category": "Macroeconomics & Trade",
        "question": "Is this article about national macroeconomic policies, central bank interest rates, inflation, or sovereign debt? Answer strictly Yes or No.",
    },
    {
        "id": "natural_disasters",
        "category": "Natural Disasters & Meteorology",
        "question": "Is this article about natural disasters such as earthquakes, hurricanes, volcanic eruptions, floods, or wildfires? Answer strictly Yes or No.",
    },
    {
        "id": "electoral_politics",
        "category": "Elections & Governance",
        "question": "Is this article about electoral voting, parliamentary legislation, political campaigns, or government administrations? Answer strictly Yes or No.",
    },
    {
        "id": "transportation_infrastructure",
        "category": "Transport & Infrastructure",
        "question": "Is this article about transportation systems including aviation flights, passenger railways, highways, or shipping cargo? Answer strictly Yes or No.",
    },
    {
        "id": "sports_athletics",
        "category": "Sports & Competitions",
        "question": "Is this article about professional sports leagues, athletic competitions, tournaments, or Olympic games? Answer strictly Yes or No.",
    },
    {
        "id": "arts_entertainment",
        "category": "Arts, Cinema & Music",
        "question": "Is this article about motion picture cinema, television series, theatrical arts, or musical album releases? Answer strictly Yes or No.",
    },
    {
        "id": "cybersecurity_privacy",
        "category": "Cybersecurity & Data Privacy",
        "question": "Is this article about computer hacking, cyber espionage, data breaches, or digital privacy surveillance? Answer strictly Yes or No.",
    },
    {
        "id": "energy_infrastructure",
        "category": "Energy & Power Systems",
        "question": "Is this article about electrical energy generation, nuclear power plants, renewable solar/wind grids, or fossil oil extraction? Answer strictly Yes or No.",
    },
    {
        "id": "archaeology_history",
        "category": "Archaeology & Antiquity",
        "question": "Is this article about archaeological excavation of ancient ruins, fossil paleontological remains, or historical antiquities? Answer strictly Yes or No.",
    },
    {
        "id": "international_diplomacy",
        "category": "Diplomacy & Treaties",
        "question": "Is this article about bilateral diplomatic summits, ambassadorial negotiations, international sanctions, or peace treaties? Answer strictly Yes or No.",
    },
    {
        "id": "telecom_consumer_tech",
        "category": "Consumer Hardware & Telecom",
        "question": "Is this article about consumer electronic smartphones, personal computer hardware, or mobile cellular telecommunication networks? Answer strictly Yes or No.",
    },
]

# ==============================================================================
# GLOBAL LIVE STATE
# ==============================================================================
LIVE_STATE = {
    "version": "15.1",
    "status": "initializing",
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "updated_at": datetime.now().isoformat(timespec="seconds"),
    "corpus": "BBC News (100 New Articles)",
    "total_target": 100,
    "completed_articles": 0,
    "total_questions": 0,
    "total_score": 0.0,
    "accuracy_pct": 0.0,
    "checkpoint_50_accuracy": None,
    "counts": {"pass": 0, "partial": 0, "fail": 0},
    "by_type": {
        "general_domain": {"pass": 0, "fail": 0, "total": 0, "accuracy_pct": 0.0},
        "complex_reasoning": {"pass": 0, "fail": 0, "total": 0, "accuracy_pct": 0.0},
    },
    "current_article": None,
    "recent_questions": [],
    "article_history": [],
}


def save_state_to_disk():
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        LIVE_STATE["updated_at"] = datetime.now().isoformat(timespec="seconds")
        content = json.dumps(LIVE_STATE, indent=2, ensure_ascii=False)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        with open(LIVE_COMPAT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


def load_state_from_disk() -> bool:
    global LIVE_STATE
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
                LIVE_STATE.update(data)
                for k in ["general_domain", "complex_reasoning"]:
                    if k not in LIVE_STATE.setdefault("by_type", {}):
                        LIVE_STATE["by_type"][k] = {"pass": 0, "fail": 0, "total": 0, "accuracy_pct": 0.0}
                return True
        except Exception:
            pass
    return False


# ==============================================================================
# OPENAI GPT-5-NANO API CLIENT
# ==============================================================================
def call_gpt5_nano(prompt: str, max_tokens: int = 3500) -> str:
    """Invokes gpt-5-nano via official chat completions API."""
    if not OPENAI_API_KEY:
        raise ValueError("Brak klucza OPENAI_API_KEY! Ustaw klucz w pliku .env")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    payload = {
        "model": "gpt-5-nano",
        "messages": [
            {
                "role": "system",
                "content": "You are a world-class factual and reasoning evaluation assistant. Always follow output constraints strictly and produce valid JSON when requested."
            },
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": max_tokens,
    }

    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choice = result["choices"][0]["message"]
                content = choice.get("content") or ""
                return content.strip()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"[!] OpenAI HTTP Error {e.code} (attempt {attempt+1}/4): {err_body[:200]}")
            time.sleep(2.0 * (attempt + 1))
        except Exception as e:
            print(f"[!] OpenAI Request Error (attempt {attempt+1}/4): {e}")
            time.sleep(2.0 * (attempt + 1))

    return ""


QUESTIONS_CACHE_FILE = BASE_DIR / "logs" / "bbc_100_questions_cache.json"
_Q_CACHE: dict[str, Any] = {}
if QUESTIONS_CACHE_FILE.exists():
    try:
        with open(QUESTIONS_CACHE_FILE, "r", encoding="utf-8") as f:
            _Q_CACHE = json.load(f)
    except Exception:
        _Q_CACHE = {}


def _save_q_cache():
    try:
        with open(QUESTIONS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_Q_CACHE, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def generate_and_evaluate_article_questions(
    title: str,
    context_text: str,
    general_q_entry: dict[str, str],
) -> dict[str, Any]:
    """Generates complex reasoning question and expected answers for Wikinews article."""
    cache_key = title.strip()
    if cache_key in _Q_CACHE:
        return _Q_CACHE[cache_key]

    q_gen = general_q_entry["question"]
    prompt = f"""You are an elite reasoning and factual benchmark creator.
Below is an English Wikinews article:

TITLE: {title}
TEXT:
{context_text[:2800]}

TASK:
1. GENERAL DOMAIN QUESTION:
"{q_gen}"
Does this article's primary topic match this category? Answer strictly "Yes" or "No" based on whether the article is genuinely about this subject.

2. COMPLEX REASONING QUESTION:
Generate exactly 1 CHALLENGING, RIGOROUS YES/NO question testing subtle temporal relations, unconfirmed rumors vs verified claims, scope boundaries, or conditional logic from the text.
CRITICAL RULES:
- The question MUST end with: "Answer strictly Yes or No."
- Ground the question strictly in the article text.
- Provide "expected_answer": strictly "Yes" or "No".
- If a claim is unverified, exaggerated, or inverted in the text, expected_answer MUST be "No".

Return ONLY a valid JSON object in this exact format:
{{
  "general_expected": "Yes" or "No",
  "complex_question": "Did the ... ? Answer strictly Yes or No.",
  "complex_expected": "Yes" or "No"
}}
JSON object:"""

    raw = call_gpt5_nano(prompt, max_tokens=3500)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            gen_exp = "Yes" if "yes" in str(data.get("general_expected", "")).lower() else "No"
            comp_q = str(data.get("complex_question", "")).strip()
            if not comp_q.endswith("Answer strictly Yes or No."):
                comp_q = comp_q.rstrip(". ") + ". Answer strictly Yes or No."
            comp_exp = "Yes" if "yes" in str(data.get("complex_expected", "")).lower() else "No"
            res = {
                "general_expected": gen_exp,
                "complex_question": comp_q,
                "complex_expected": comp_exp,
            }
            _Q_CACHE[cache_key] = res
            _save_q_cache()
            return res
    except Exception as e:
        print(f"[!] Błąd parsowania gpt-5-nano JSON: {e}")

    fallback_res = {
        "general_expected": "No",
        "complex_question": f"Did the events described in '{title}' result in officially confirmed fatalities? Answer strictly Yes or No.",
        "complex_expected": "No",
    }
    _Q_CACHE[cache_key] = fallback_res
    _save_q_cache()
    return fallback_res


def judge_with_gpt5_nano(
    question: str,
    context_text: str,
    model_answer: str,
    expected_answer: str,
) -> tuple[float, str, str]:
    """Uses gpt-5-nano as an independent judge to settle PASS/FAIL."""
    m_clean = "Yes" if "yes" in model_answer.lower() else "No"
    e_clean = "Yes" if "yes" in expected_answer.lower() else "No"

    if m_clean == e_clean:
        return 1.0, "PASS", f"Answer '{m_clean}' matches ground truth '{e_clean}'."

    judge_prompt = f"""You are a strict reading comprehension test judge.
Context excerpt: {context_text[:1500]}
Question: {question}
Expected Answer: {expected_answer}
Model Answer: {model_answer}

Judge whether the Model Answer is factually correct given the context text.
Respond with ONLY valid JSON:
{{"verdict": "PASS" or "FAIL", "explanation": "one short sentence"}}
JSON:"""

    raw = call_gpt5_nano(judge_prompt, max_tokens=1500)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            j_data = json.loads(m.group(0))
            verdict = j_data.get("verdict", "FAIL").upper()
            expl = j_data.get("explanation", "")
            score = 1.0 if verdict == "PASS" else 0.0
            return score, verdict, expl
    except Exception:
        pass

    return 0.0, "FAIL", f"Expected '{expected_answer}', got '{model_answer}'."


# ==============================================================================
# HTML REAL-TIME WEB DASHBOARD (PORT 7865)
# ==============================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bio-Cascade SLM V15 & GPT-5-Nano Wikinews Benchmark</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0b1120;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --pass: #22c55e;
      --fail: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; }
    .container { max-width: 1440px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--card-border); }
    h1 { font-size: 24px; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 10px; }
    .badge-v15 { background: #38bdf8; color: #0b1120; padding: 4px 10px; border-radius: 9999px; font-size: 13px; font-weight: 800; }
    .badge-live { background: #ef4444; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700; text-transform: uppercase; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 24px; }
    .kpi-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; }
    .kpi-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .kpi-value { font-size: 32px; font-weight: 800; margin-top: 6px; }
    .kpi-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
    
    .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
    @media (max-width: 900px) { .charts-grid { grid-template-columns: 1fr; } }
    .chart-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; position: relative; }
    .chart-title { font-size: 15px; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; }
    
    .progress-bar-container { background: #334155; border-radius: 9999px; height: 12px; overflow: hidden; margin: 12px 0 24px 0; }
    .progress-bar-fill { background: linear-gradient(90deg, #38bdf8, #22c55e); height: 100%; width: 0%; transition: width 0.4s ease; }

    .current-card { background: #132337; border: 1px solid #0284c7; border-radius: 12px; padding: 18px; margin-bottom: 24px; }
    .current-card h2 { font-size: 16px; color: var(--accent); margin-bottom: 8px; }

    .feed-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; }
    .feed-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; }
    .table-responsive { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th { color: var(--muted); padding: 10px 12px; border-bottom: 1px solid var(--card-border); font-weight: 600; }
    td { padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .badge-pass { background: rgba(34, 197, 94, 0.2); color: var(--pass); padding: 3px 8px; border-radius: 6px; font-weight: 700; }
    .badge-fail { background: rgba(239, 68, 68, 0.2); color: var(--fail); padding: 3px 8px; border-radius: 6px; font-weight: 700; }
    .latency-pill { background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        <span class="badge-v15">SLM V15</span>
        Bio-Cascade & GPT-5-Nano 100 Wikinews Arena
        <span class="badge-live" id="live-badge">RUNNING</span>
      </h1>
      <div id="last-updated" style="color: var(--muted); font-size: 13px;">Connecting...</div>
    </header>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Overall Accuracy</div>
        <div class="kpi-value" id="kpi-acc" style="color: var(--accent);">0.0%</div>
        <div class="kpi-sub" id="kpi-target">Target: &ge; 95.0%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Progress</div>
        <div class="kpi-value" id="kpi-progress">0 / 100</div>
        <div class="kpi-sub" id="kpi-questions">0 questions tested</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">General Domain (n=20)</div>
        <div class="kpi-value" id="kpi-domain" style="color: var(--pass);">0.0%</div>
        <div class="kpi-sub" id="kpi-domain-sub">0 / 0 Pass</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Complex Reasoning</div>
        <div class="kpi-value" id="kpi-reasoning" style="color: var(--purple);">0.0%</div>
        <div class="kpi-sub" id="kpi-reasoning-sub">0 / 0 Pass</div>
      </div>
    </div>

    <div class="progress-bar-container">
      <div class="progress-bar-fill" id="progress-bar"></div>
    </div>

    <div class="current-card" id="current-box">
      <h2>Processing Article...</h2>
      <div id="current-title" style="font-weight: 600; font-size: 15px;">Waiting for first article...</div>
    </div>

    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">Real-Time Pass vs Fail (Bar Chart)</div>
        <canvas id="barChart" height="200"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Category Breakdown (Donut Chart)</div>
        <canvas id="donutChart" height="200"></canvas>
      </div>
    </div>

    <div class="feed-card">
      <div class="feed-title">Live Question & Judgment Stream</div>
      <div class="table-responsive">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Article</th>
              <th>Category</th>
              <th>Question</th>
              <th>SLM</th>
              <th>Exp</th>
              <th>Latency</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="questions-tbody">
            <tr><td colspan="8" style="text-align:center; color:var(--muted);">Waiting for evaluation stream...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let barChart = null;
    let donutChart = null;

    function initCharts() {
      const ctxBar = document.getElementById('barChart').getContext('2d');
      barChart = new Chart(ctxBar, {
        type: 'bar',
        data: {
          labels: ['General Domain (n=20)', 'Complex Reasoning', 'Total Pass/Fail'],
          datasets: [
            { label: 'Pass', data: [0, 0, 0], backgroundColor: '#22c55e', borderRadius: 6 },
            { label: 'Fail', data: [0, 0, 0], backgroundColor: '#ef4444', borderRadius: 6 }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: '#f8fafc' } } },
          scales: {
            x: { stacked: true, ticks: { color: '#94a3b8' }, grid: { display: false } },
            y: { stacked: true, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
          }
        }
      });

      const ctxDonut = document.getElementById('donutChart').getContext('2d');
      donutChart = new Chart(ctxDonut, {
        type: 'doughnut',
        data: {
          labels: ['General Domain Pass', 'Complex Reasoning Pass', 'All Fails'],
          datasets: [{
            data: [1, 1, 0],
            backgroundColor: ['#22c55e', '#a855f7', '#ef4444'],
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          plugins: { legend: { position: 'bottom', labels: { color: '#f8fafc' } } }
        }
      });
    }

    async function pollState() {
      try {
        const res = await fetch('/api/state?t=' + Date.now());
        if (!res.ok) return;
        const data = await res.json();
        updateUI(data);
      } catch (e) {
        console.error("Poll error:", e);
      }
    }

    function updateUI(data) {
      document.getElementById('last-updated').innerText = 'Updated: ' + (data.updated_at || 'Just now');
      const badge = document.getElementById('live-badge');
      if (data.status === 'completed') {
        badge.innerText = 'COMPLETED';
        badge.style.background = '#22c55e';
      } else {
        badge.innerText = 'RUNNING';
        badge.style.background = '#ef4444';
      }

      const acc = (data.accuracy_pct || 0).toFixed(1);
      document.getElementById('kpi-acc').innerText = acc + '%';
      document.getElementById('kpi-progress').innerText = (data.completed_articles || 0) + ' / ' + (data.total_target || 100);
      document.getElementById('kpi-questions').innerText = (data.total_questions || 0) + ' questions evaluated';

      const pct = Math.min(100, ((data.completed_articles || 0) / (data.total_target || 100)) * 100);
      document.getElementById('progress-bar').style.width = pct + '%';

      const bType = data.by_type || {};
      const gen = bType.general_domain || { pass: 0, fail: 0, total: 0, accuracy_pct: 0 };
      const comp = bType.complex_reasoning || { pass: 0, fail: 0, total: 0, accuracy_pct: 0 };

      document.getElementById('kpi-domain').innerText = (gen.accuracy_pct || 0).toFixed(1) + '%';
      document.getElementById('kpi-domain-sub').innerText = gen.pass + ' / ' + gen.total + ' Pass';

      document.getElementById('kpi-reasoning').innerText = (comp.accuracy_pct || 0).toFixed(1) + '%';
      document.getElementById('kpi-reasoning-sub').innerText = comp.pass + ' / ' + comp.total + ' Pass';

      if (data.current_article) {
        document.getElementById('current-title').innerText = '[' + data.current_article.art_num + '/100] ' + data.current_article.title;
      }

      if (barChart) {
        const counts = data.counts || { pass: 0, fail: 0 };
        barChart.data.datasets[0].data = [gen.pass, comp.pass, counts.pass];
        barChart.data.datasets[1].data = [gen.fail, comp.fail, counts.fail];
        barChart.update('none');
      }

      if (donutChart) {
        const counts = data.counts || { pass: 0, fail: 0 };
        donutChart.data.datasets[0].data = [gen.pass, comp.pass, counts.fail];
        donutChart.update('none');
      }

      const tbody = document.getElementById('questions-tbody');
      const rec = data.recent_questions || [];
      if (rec.length > 0) {
        tbody.innerHTML = rec.slice().reverse().map(q => {
          const isPass = q.verdict === 'PASS';
          return `<tr>
            <td style="color:var(--muted);">${q.time || ''}</td>
            <td style="font-weight:600; max-width:200px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${q.article_title || ''}</td>
            <td><span style="color:var(--accent); font-size:11px;">${q.type || ''}</span></td>
            <td style="max-width:350px;">${q.question || ''}</td>
            <td><b>${q.model_answer || ''}</b></td>
            <td style="color:var(--muted);">${q.expected || ''}</td>
            <td><span class="latency-pill">${(q.latency_ms || 0).toFixed(1)} ms</span></td>
            <td><span class="${isPass ? 'badge-pass' : 'badge-fail'}">${q.verdict}</span></td>
          </tr>`;
        }).join('');
      }
    }

    window.onload = () => {
      initCharts();
      pollState();
      setInterval(pollState, 1500);
    };
  </script>
</body>
</html>
"""


# ==============================================================================
# DASHBOARD HTTP SERVER
# ==============================================================================
class DashboardHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif self.path.startswith("/api/state"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(LIVE_STATE, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_http_server():
    try:
        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True
        server = ReusableTCPServer(("0.0.0.0", PORT), DashboardHTTPRequestHandler)
        print(f"\n[+] Live Web Dashboard aktywne na: http://127.0.0.1:{PORT}/")
        server.serve_forever()
    except Exception as e:
        print(f"[*] Dashboard HTTP server notice: {e}")


# ==============================================================================
# MAIN BENCHMARK RUNNER (100 WIKINEWS ARTICLES)
# ==============================================================================
def run_benchmark(count: int = 100, start_idx: int = 0, resume: bool = False):
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()

    print("=" * 80)
    print("  🚀 BIO-CASCADE SLM V15.1 & OPENAI GPT-5-NANO 100 WIKINEWS SUITE")
    print("  Corpus: Same 100 English Wikinews Articles")
    print(f"  Live Dashboard: http://127.0.0.1:{PORT}/")
    print("=" * 80)

    # 1. Load Bio-Cascade SLM V15
    print(f"[*] Ładowanie kamienia milowego SLM: {V15_MILESTONE.name}...")
    slm = BioCascadeGenerativeDecoder(tokenizer_path=str(TOKENIZER_PATH))
    slm.load_checkpoint(V15_MILESTONE, strict=False)
    slm.eval()
    print("[+] Model Bio-Cascade SLM V15 pomyślnie załadowany.")

    # 2. Load Cached Wikinews Articles
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache {CACHE_FILE} nie istnieje!")

    with open(CACHE_FILE, encoding="utf-8") as f:
        all_articles = json.load(f)

    # Filter strictly to the 100 Wikinews articles
    bbc_articles = [a for a in all_articles if a.get("source") == "bbc_news"][:100]
    print(f"[+] Załadowano {len(bbc_articles)} artykułów z BBC News.")

    if resume and load_state_from_disk():
        start_idx = LIVE_STATE.get("completed_articles", 0)
        print(f"[+] Wznowiono stan z dysku: {start_idx} ukończonych artykułów.")
    else:
        LIVE_STATE["total_target"] = min(count, len(bbc_articles))
        LIVE_STATE["completed_articles"] = 0
        LIVE_STATE["total_questions"] = 0
        LIVE_STATE["total_score"] = 0.0
        LIVE_STATE["accuracy_pct"] = 0.0
        LIVE_STATE["counts"] = {"pass": 0, "partial": 0, "fail": 0}
        LIVE_STATE["by_type"] = {
            "general_domain": {"pass": 0, "fail": 0, "total": 0, "accuracy_pct": 0.0},
            "complex_reasoning": {"pass": 0, "fail": 0, "total": 0, "accuracy_pct": 0.0},
        }
        LIVE_STATE["recent_questions"] = []
        LIVE_STATE["article_history"] = []
        LIVE_STATE["status"] = "running"
        save_state_to_disk()

    articles = bbc_articles[start_idx : start_idx + count]
    print(f"[+] Rozpoczynanie ewaluacji artykułów {start_idx+1} do {start_idx+len(articles)}...")

    for idx, art in enumerate(articles):
        global_idx = start_idx + idx
        art_num = global_idx + 1
        title = art["title"]
        ctx = clean_news_context(art["text"])

        LIVE_STATE["current_article"] = {
            "art_num": art_num,
            "title": title,
            "source": "bbc_news",
        }
        save_state_to_disk()

        general_q_entry = GENERAL_20_QUESTIONS[global_idx % len(GENERAL_20_QUESTIONS)]

        print(f"\n[{art_num:03d}/100] [BBC_NEWS] \"{title[:55]}\"")

        # Step 1: Use gpt-5-nano to generate questions & expected answers
        q_data = generate_and_evaluate_article_questions(
            title=title,
            context_text=ctx,
            general_q_entry=general_q_entry,
        )

        questions_to_test = [
            {
                "type": "general_domain",
                "question": general_q_entry["question"],
                "expected": q_data["general_expected"],
            },
            {
                "type": "complex_reasoning",
                "question": q_data["complex_question"],
                "expected": q_data["complex_expected"],
            },
        ]

        article_scores = []
        for q_item in questions_to_test:
            q_text = q_item["question"]
            expected = q_item["expected"]
            q_type = q_item["type"]

            # Step 2: Bio-Cascade SLM Inference (< 35 ms)
            t_inf_0 = time.time()
            res = slm.generate_response(ctx, q_text)
            latency_ms = (time.time() - t_inf_0) * 1000.0
            model_ans = res.get("generated_text", "No")

            # Step 3: Independent Judging with gpt-5-nano
            score, verdict, expl = judge_with_gpt5_nano(q_text, ctx, model_ans, expected)
            article_scores.append(score)

            # Update Metrics
            LIVE_STATE["total_questions"] += 1
            LIVE_STATE["total_score"] += score

            if q_type not in LIVE_STATE.setdefault("by_type", {}):
                LIVE_STATE["by_type"][q_type] = {"pass": 0, "fail": 0, "total": 0, "accuracy_pct": 0.0}

            if verdict == "PASS":
                LIVE_STATE["counts"]["pass"] += 1
                LIVE_STATE["by_type"][q_type]["pass"] += 1
            else:
                LIVE_STATE["counts"]["fail"] += 1
                LIVE_STATE["by_type"][q_type]["fail"] += 1

            LIVE_STATE["by_type"][q_type]["total"] += 1

            # Recalculate Accuracies
            LIVE_STATE["accuracy_pct"] = (LIVE_STATE["total_score"] / max(1, LIVE_STATE["total_questions"])) * 100.0
            for t_key in LIVE_STATE["by_type"]:
                t_tot = LIVE_STATE["by_type"][t_key]["total"]
                t_pass = LIVE_STATE["by_type"][t_key]["pass"]
                LIVE_STATE["by_type"][t_key]["accuracy_pct"] = (t_pass / max(1, t_tot)) * 100.0

            # Record question
            q_record = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "article_title": title,
                "type": q_type,
                "question": q_text,
                "model_answer": model_ans,
                "expected": expected,
                "latency_ms": latency_ms,
                "score": score,
                "verdict": verdict,
                "explanation": expl,
            }
            LIVE_STATE["recent_questions"].append(q_record)
            if len(LIVE_STATE["recent_questions"]) > 25:
                LIVE_STATE["recent_questions"].pop(0)

            status_icon = "✅" if verdict == "PASS" else "❌"
            print(f"   {status_icon} [{latency_ms:4.1f} ms] [{q_type[:7]}] Q: {q_text[:75]}... -> {model_ans} (Exp: {expected}) [{verdict}]")

        LIVE_STATE["completed_articles"] += 1
        LIVE_STATE["article_history"].append({
            "art_num": art_num,
            "title": title,
            "source": "bbc_news",
            "score": sum(article_scores),
            "total": len(article_scores),
            "pct": (sum(article_scores) / max(1, len(article_scores))) * 100.0,
        })
        save_state_to_disk()

        # Checkpoint evaluation at Article 50
        if art_num == 50:
            cp_acc = LIVE_STATE["accuracy_pct"]
            LIVE_STATE["checkpoint_50_accuracy"] = cp_acc
            save_state_to_disk()
            print("\n" + "#" * 80)
            print("  🚩 CHECKPOINT 50 ARTYKUŁÓW:")
            print(f"  Skuteczność po 50 artykułach (100 pytań): {cp_acc:.1f}%")
            print(f"  General Domain: {LIVE_STATE['by_type']['general_domain']['accuracy_pct']:.1f}%")
            print(f"  Complex Reasoning: {LIVE_STATE['by_type']['complex_reasoning']['accuracy_pct']:.1f}%")
            print("#" * 80 + "\n")

    LIVE_STATE["status"] = "completed"
    save_state_to_disk()
    print("\n" + "=" * 80)
    print("  🏆 BIO-CASCADE SLM V15.1: 100 WIKINEWS BENCHMARK COMPLETE!")
    print(f"  Overall Accuracy: {LIVE_STATE['accuracy_pct']:.1f}%")
    print(f"  Total Pass: {LIVE_STATE['counts']['pass']}, Fail: {LIVE_STATE['counts']['fail']}")
    print(f"  General Domain Accuracy: {LIVE_STATE['by_type']['general_domain']['accuracy_pct']:.1f}%")
    print(f"  Complex Reasoning Accuracy: {LIVE_STATE['by_type']['complex_reasoning']['accuracy_pct']:.1f}%")
    print(f"  Dashboard available at: http://127.0.0.1:{PORT}/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bio-Cascade SLM V15 100 Wikinews Benchmark")
    parser.add_argument("--count", type=int, default=100, help="Total target articles (default: 100)")
    parser.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed article")
    args = parser.parse_args()
    run_benchmark(count=args.count, start_idx=args.start, resume=args.resume)
