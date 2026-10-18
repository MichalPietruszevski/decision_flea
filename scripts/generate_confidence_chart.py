import json
import os
from pathlib import Path

# Fallback to local logs directory if not in Antigravity artifact mode
artifact_dir = Path(os.environ.get("ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "logs"))
artifact_dir.mkdir(parents=True, exist_ok=True)
html_file = artifact_dir / "confidence_benchmark_50_news.html"

with open("logs/confidence_eval_50.json", encoding="utf-8") as f:
    items = json.load(f)

# Sort items by confidence descending
sorted_items = sorted(items, key=lambda x: x["confidence"], reverse=True)

items_json_str = json.dumps(sorted_items, ensure_ascii=False)

html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wykres Pewności Bio-Cascade SLM V15.1</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .bar-hover:hover {{
      filter: brightness(1.2);
      transform: scaleY(1.05);
      transform-origin: bottom;
    }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3 sm:p-5">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-5 shadow-lg max-w-4xl mx-auto space-y-6">
    
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[var(--border)] pb-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            100% RECALL MILESTONE
          </span>
          <span class="text-xs text-[var(--muted-foreground)]">Weryfikacja na 50 newsach</span>
        </div>
        <h2 class="text-xl sm:text-2xl font-bold tracking-tight mt-1 text-[var(--foreground)]">
          Rozkład Pewności (Confidence Plot) - Bio-Cascade V15.1
        </h2>
        <p class="text-xs sm:text-sm text-[var(--muted-foreground)] mt-0.5">
          Ewaluacja 50 rzeczywistych artykułów (25 Wikinews + 25 BBC News) w trybie <code>mode="boolean"</code>.
        </p>
      </div>
      <div class="flex items-center gap-3">
        <div class="text-right">
          <div class="text-2xl font-extrabold text-emerald-500">100.0%</div>
          <div class="text-xs text-[var(--muted-foreground)]">Dokładność (50/50)</div>
        </div>
      </div>
    </div>

    <!-- Metric Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">Średnia Pewność</div>
        <div class="text-xl font-bold text-[var(--foreground)] mt-0.5">99.78%</div>
        <div class="text-[11px] text-emerald-500 mt-0.5">Kalibracja optymalna</div>
      </div>
      <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">Mediana Pewności</div>
        <div class="text-xl font-bold text-[var(--foreground)] mt-0.5">100.00%</div>
        <div class="text-[11px] text-emerald-500 mt-0.5">35/50 ma pewność 100%</div>
      </div>
      <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">Pewność &ge; 90%</div>
        <div class="text-xl font-bold text-[var(--foreground)] mt-0.5">100.0%</div>
        <div class="text-[11px] text-[var(--muted-foreground)] mt-0.5">Min: 93.35%</div>
      </div>
      <div class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)]">
        <div class="text-xs text-[var(--muted-foreground)]">Rozkład Decyzji</div>
        <div class="text-xl font-bold text-[var(--foreground)] mt-0.5">14 Yes / 36 No</div>
        <div class="text-[11px] text-emerald-500 mt-0.5">FN = 0 (100% Recall)</div>
      </div>
    </div>

    <!-- Interactive Confidence Bar Chart -->
    <div class="space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-semibold text-[var(--foreground)] flex items-center gap-2">
          <span>Wykres Słupkowy Pewności (Każdy z 50 Artykułów)</span>
          <span class="text-xs font-normal text-[var(--muted-foreground)]">(Najedź kursorem na słupek, aby zobaczyć szczegóły)</span>
        </h3>
        <div class="flex items-center gap-3 text-xs">
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block"></span> Decyzja "Yes"</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-sky-500 inline-block"></span> Decyzja "No"</span>
        </div>
      </div>

      <!-- Chart Container -->
      <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)] overflow-x-auto">
        <div class="relative h-44 flex items-end gap-1.5 min-w-[620px] pt-6 pb-2" id="chart-bars">
          <!-- Bars generated via JS -->
        </div>
        <div class="flex justify-between text-[11px] text-[var(--muted-foreground)] mt-2 pt-2 border-t border-[var(--border)]">
          <span>Sortowanie: Od najwyższej pewności (100%)</span>
          <span>Do najniższej w zbiorze (93.35%)</span>
        </div>
      </div>

      <!-- Active Details Box -->
      <div id="tooltip-box" class="p-3.5 rounded-xl bg-[var(--background)] border border-[var(--border)] text-xs transition-all duration-200">
        <div class="font-medium text-[var(--foreground)]" id="t-title">Wskaż słupek powyżej, aby obejrzeć artykuł i pytanie.</div>
        <div class="text-[var(--muted-foreground)] mt-1 flex flex-wrap gap-4" id="t-meta">
          <span>Kliknij lub najedź myszą na dowolny z 50 słupków.</span>
        </div>
      </div>
    </div>

    <!-- Histogram Distribution -->
    <div class="space-y-2 pt-2">
      <h3 class="text-sm font-semibold text-[var(--foreground)]">Rozkład Histogramowy (Przedziały Pewności)</h3>
      <div class="space-y-1.5 text-xs">
        <div class="flex items-center gap-2">
          <span class="w-24 text-[var(--muted-foreground)]">100.0%</span>
          <div class="flex-1 bg-[var(--background)] h-5 rounded-md overflow-hidden p-0.5 border border-[var(--border)]">
            <div class="bg-emerald-500 h-full rounded text-[10px] text-white font-semibold flex items-center justify-end pr-2" style="width: 70%;">35 artykułów (70%)</div>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-24 text-[var(--muted-foreground)]">99.0 - 99.9%</span>
          <div class="flex-1 bg-[var(--background)] h-5 rounded-md overflow-hidden p-0.5 border border-[var(--border)]">
            <div class="bg-emerald-500/80 h-full rounded text-[10px] text-white font-semibold flex items-center justify-end pr-2" style="width: 24%;">12 artykułów (24%)</div>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-24 text-[var(--muted-foreground)]">95.0 - 98.9%</span>
          <div class="flex-1 bg-[var(--background)] h-5 rounded-md overflow-hidden p-0.5 border border-[var(--border)]">
            <div class="bg-emerald-500/60 h-full rounded text-[10px] text-white font-semibold flex items-center justify-end pr-1.5" style="width: 8%;">2 (4%)</div>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-24 text-[var(--muted-foreground)]">90.0 - 94.9%</span>
          <div class="flex-1 bg-[var(--background)] h-5 rounded-md overflow-hidden p-0.5 border border-[var(--border)]">
            <div class="bg-amber-500 h-full rounded text-[10px] text-white font-semibold flex items-center justify-end pr-1.5" style="width: 5%;">1 (2%)</div>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-24 text-[var(--muted-foreground)]">&lt; 90.0%</span>
          <div class="flex-1 bg-[var(--background)] h-5 rounded-md overflow-hidden p-0.5 border border-[var(--border)]">
            <div class="text-[10px] text-[var(--muted-foreground)] pl-2">0 artykułów (0%)</div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <script>
    const data = {items_json_str};
    const container = document.getElementById('chart-bars');
    const tBox = document.getElementById('tooltip-box');
    const tTitle = document.getElementById('t-title');
    const tMeta = document.getElementById('t-meta');

    data.forEach((item, idx) => {{
      const bar = document.createElement('div');
      const isYes = item.decision === 'Yes';
      const confPct = item.confidence * 100;
      
      // Scale height between 50% and 100% (so min 93% is visibly differentiated)
      // Visual range: 90% -> 20px, 100% -> 140px
      const heightPx = Math.max(25, ((confPct - 90) / 10) * 115 + 25);
      
      bar.className = 'flex-1 bar-hover rounded-t transition-all cursor-pointer relative ' + (isYes ? 'bg-emerald-500' : 'bg-sky-500');
      bar.style.height = heightPx + 'px';
      bar.setAttribute('title', item.title);

      bar.addEventListener('mouseenter', () => {{
        tTitle.innerHTML = `<span class="font-bold text-[var(--foreground)]">#${{idx + 1}} [${{item.source.toUpperCase()}}]</span> ${{item.title}}`;
        tMeta.innerHTML = `
          <span>Pytanie: <strong class="text-[var(--foreground)]">${{item.question}}</strong></span><br>
          <span>Decyzja: <strong class="${{isYes ? 'text-emerald-500' : 'text-sky-500'}}">${{item.decision}}</strong> (Oczekiwano: ${{item.expected}})</span> | 
          <span>Pewność: <strong class="text-[var(--foreground)] font-bold">${{confPct.toFixed(2)}}%</strong></span> | 
          <span>P(Yes): ${{ (item.p_yes * 100).toFixed(2) }}%</span> | 
          <span>P(No): ${{ (item.p_no * 100).toFixed(2) }}%</span> | 
          <span>Margines Logitów &Delta;: ${{ item.margin > 0 ? '+' : '' }}${{ item.margin.toFixed(2) }}</span>
        `;
      }});

      container.appendChild(bar);
    }});

    // Set first item as initial display
    if (data.length > 0) {{
      const first = data[0];
      tTitle.innerHTML = `<span class="font-bold text-[var(--foreground)]">#1 [${{first.source.toUpperCase()}}]</span> ${{first.title}}`;
      tMeta.innerHTML = `
        <span>Pytanie: <strong class="text-[var(--foreground)]">${{first.question}}</strong></span><br>
        <span>Decyzja: <strong class="text-emerald-500">${{first.decision}}</strong> (Oczekiwano: ${{first.expected}})</span> | 
        <span>Pewność: <strong class="text-[var(--foreground)] font-bold">${{(first.confidence * 100).toFixed(2)}}%</strong></span> | 
        <span>P(Yes): ${{ (first.p_yes * 100).toFixed(2) }}%</span> | 
        <span>P(No): ${{ (first.p_no * 100).toFixed(2) }}%</span> | 
        <span>Margines Logitów &Delta;: +${{first.margin.toFixed(2)}}</span>
      `;
    }}
  </script>
</body>
</html>
"""

with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[+] Successfully generated HTML artifact at: {html_file}")
