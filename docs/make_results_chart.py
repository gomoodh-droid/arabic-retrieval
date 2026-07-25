#!/usr/bin/env python3
"""
مولّد الرسم البياني للنتائج بصيغة SVG ناصعة بدقة عالية.
"""
from pathlib import Path

SVG_CONTENT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 420" width="100%" height="100%" style="background:#0F172A; font-family: system-ui, -apple-system, sans-serif;">
  <style>
    .title { font-size: 16px; font-weight: bold; fill: #F8FAFC; }
    .label { font-size: 12px; fill: #94A3B8; }
    .bar-txt { font-size: 12px; font-weight: bold; fill: #FFFFFF; }
    .legend-txt { font-size: 12px; fill: #CBD5E1; }
  </style>

  <text x="425" y="35" text-anchor="middle" class="title">مقارنة أداء استراتيجيات الاسترجاع حسب حجم المتن (Recall@1 Benchmark)</text>

  <!-- Y Axis Grid -->
  <line x1="80" y1="320" x2="780" y2="320" stroke="#334155" stroke-width="1"/>
  <line x1="80" y1="250" x2="780" y2="250" stroke="#334155" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="80" y1="180" x2="780" y2="180" stroke="#334155" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="80" y1="110" x2="780" y2="110" stroke="#334155" stroke-width="1" stroke-dasharray="3 3"/>

  <text x="70" y="325" text-anchor="end" class="label">0.0</text>
  <text x="70" y="255" text-anchor="end" class="label">0.15</text>
  <text x="70" y="185" text-anchor="end" class="label">0.30</text>
  <text x="70" y="115" text-anchor="end" class="label">0.45</text>

  <!-- Group 1: 500 Docs -->
  <g transform="translate(110,0)">
    <rect x="0" y="222" width="22" height="98" fill="#64748B" rx="3"/>
    <rect x="26" y="171" width="22" height="149" fill="#10B981" rx="3"/>
    <rect x="52" y="161" width="22" height="159" fill="#F59E0B" rx="3"/>
    <rect x="78" y="146" width="22" height="174" fill="#8B5CF6" rx="3"/>
    <text x="50" y="345" text-anchor="middle" class="label">500 مستند</text>
  </g>

  <!-- Group 2: 4,000 Docs -->
  <g transform="translate(340,0)">
    <rect x="0" y="255" width="22" height="65" fill="#64748B" rx="3"/>
    <rect x="26" y="187" width="22" height="133" fill="#10B981" rx="3"/>
    <rect x="52" y="209" width="22" height="111" fill="#F59E0B" rx="3"/>
    <rect x="78" y="165" width="22" height="155" fill="#8B5CF6" rx="3"/>
    <text x="50" y="345" text-anchor="middle" class="label">4,000 مستند</text>
  </g>

  <!-- Group 3: 16,000 Docs -->
  <g transform="translate(570,0)">
    <rect x="0" y="266" width="22" height="54" fill="#64748B" rx="3"/>
    <rect x="26" y="196" width="22" height="124" fill="#10B981" rx="3"/>
    <rect x="52" y="237" width="22" height="83" fill="#F59E0B" rx="3"/>
    <rect x="78" y="177" width="22" height="143" fill="#8B5CF6" rx="3"/>
    <text x="50" y="345" text-anchor="middle" class="label">16,000 مستند</text>
  </g>

  <!-- Legend -->
  <g transform="translate(180,385)">
    <rect x="0" y="0" width="14" height="14" fill="#64748B" rx="2"/>
    <text x="22" y="12" class="legend-txt">BM25 الخام</text>

    <rect x="130" y="0" width="14" height="14" fill="#10B981" rx="2"/>
    <text x="152" y="12" class="legend-txt">التجذيع الخفيف (Stemming)</text>

    <rect x="330" y="0" width="14" height="14" fill="#F59E0B" rx="2"/>
    <text x="352" y="12" class="legend-txt">المقاطع الحرفية (N-Grams)</text>

    <rect x="520" y="0" width="14" height="14" fill="#8B5CF6" rx="2"/>
    <text x="542" y="12" class="legend-txt">المزيج الموصى به (Hybrid)</text>
  </g>
</svg>
"""

def generate():
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    svg_path = out_dir / "results_chart.svg"
    svg_path.write_text(SVG_CONTENT, encoding="utf-8")
    print(f"Generated {svg_path}")

if __name__ == "__main__":
    generate()
