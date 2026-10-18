"""50-Question Comprehensive Benchmark on the History of Poland (Bio-Cascade V13.0).

Tests:
1. Persons / Rulers (Q01-Q10)
2. Dates / Years / Numbers (Q11-Q20)
3. Entities / Dynasties / Concepts (Q21-Q30)
4. Boolean Verification (Yes / No) (Q31-Q40)
5. Honest Refusal Gate (Unmentioned facts) (Q41-Q45)
6. Typos & Diacritic Robustness (Q46-Q49)
7. Grounded Paragraph Synthesis (Q50)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import torch
from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

CONTEXT = (
    "The history of Poland spans over a thousand years, from medieval tribes, Christianization and monarchy; "
    "through Poland's Golden Age, expansionism and becoming one of the largest European powers; to its collapse and "
    "partitions, two world wars, communism, and the restoration of democracy. The roots of Polish history can be "
    "traced to ancient times, when the territory of present-day Poland was inhabited by diverse ethnic groups, "
    "including Celts, Scythians, Sarmatians, Slavs, Balts and Germanic peoples. However, it was the West Slavic "
    "Lechites, the closest ancestors of ethnic Poles, who established permanent settlements during the Early Middle "
    "Ages. The Lechitic Western Polans, a tribe whose name denotes \"people living in open fields\", dominated the "
    "region and gave Poland—which lies in the North-Central European Plain—its name. The first ruling dynasty, the "
    "Piasts, emerged in the 10th century AD. Duke Mieszko I, regarded as the creator of Polish statehood, adopted "
    "Western Christianity in 966 AD. Mieszko's dominion was formally reconstituted as a medieval kingdom in 1025 "
    "by his son Bolesław I the Brave, known for his military expansions. The most successful and the last Piast "
    "monarch, Casimir III the Great, presided over a period of economic prosperity and territorial aggrandizement "
    "before his death in 1370 without male heirs. The period of the Jagiellonian dynasty in the 14th–16th centuries "
    "brought close ties with the Lithuania, a cultural Renaissance in Poland and continued territorial expansion as "
    "well as Polonization that culminated in the establishment of the Polish–Lithuanian Commonwealth in 1569, one of "
    "Europe's great powers. The Commonwealth was able to sustain centuries-long prosperity, while its political "
    "system matured as a unique noble democracy with an elective monarchy. From the mid-17th century, however, the "
    "huge state entered a period of decline caused by devastating wars and the deterioration of its political system. "
    "Significant internal reforms were introduced in the late 18th century, such as Europe's first Constitution of "
    "3 May 1791. The existence of the Commonwealth ended in 1795 after a series of invasions and partitions carried "
    "out by the Russian Empire, the Kingdom of Prussia and the Habsburg monarchy. From 1795 until 1918, no truly "
    "independent Polish state existed, although strong resistance movements operated. The opportunity to regain "
    "sovereignty only materialized after World War I, when the three partitioning powers were fatally weakened in the "
    "wake of war and revolution. The Second Polish Republic was established in 1918 and existed as an independent "
    "state until 1939, when Nazi Germany and the Soviet Union invaded Poland, marking the beginning of World War II. "
    "Millions of Polish citizens of different faiths or identities perished under Nazi occupation between 1939 and "
    "1945 through planned genocide and extermination. A Polish government-in-exile functioned throughout the war, "
    "and the Poles contributed to the Allied victory through participation in military campaigns on both the eastern "
    "and western fronts. The westward advances of the Soviet Red Army in 1944 and 1945 compelled Nazi Germany's "
    "forces to retreat from Poland, which led to the establishment of a satellite communist country, known from 1952 "
    "as the Polish People's Republic. The territorial adjustments mandated by the Allies at the end of World War II "
    "in 1945 shifted Poland's geographic centre of gravity towards the west, and the re-defined Polish lands largely "
    "lost their historic multi-ethnic character. By the late 1980s, the Polish reform movement Solidarity contributed "
    "to a peaceful transition from a planned communist economy to capitalism and a liberal parliamentary democracy. "
    "This process resulted in the creation of the modern Polish state, the Third Polish Republic, founded in 1989."
)

QUESTIONS = [
    # Cat 1: Persons / Rulers
    (1, "Who is regarded as the creator of Polish statehood?", ["Duke Mieszko I", "Mieszko I", "Mieszko"], "word", "Persons"),
    (2, "Who reconstituted Mieszko's dominion as a medieval kingdom in 1025?", ["Bolesław I the Brave", "Bolesław I", "Bolesław"], "word", "Persons"),
    (3, "Who was the last Piast monarch?", ["Casimir III the Great", "Casimir III", "Casimir"], "word", "Persons"),
    (4, "Who adopted Western Christianity in 966 AD?", ["Duke Mieszko I", "Mieszko I", "Mieszko"], "word", "Persons"),
    (5, "Who was known for his military expansions after Duke Mieszko I?", ["Bolesław I the Brave", "Bolesław I", "Bolesław"], "word", "Persons"),
    (6, "Who presided over a period of economic prosperity before dying in 1370?", ["Casimir III the Great", "Casimir III", "Casimir"], "word", "Persons"),
    (7, "Which monarch died in 1370 without male heirs?", ["Casimir III the Great", "Casimir III", "Casimir"], "word", "Persons"),
    (8, "Who were the closest ancestors of ethnic Poles?", ["West Slavic Lechites", "Lechites"], "word", "Persons"),
    (9, "Who contributed to the Allied victory on both eastern and western fronts?", ["Poles", "Polish citizens"], "word", "Persons"),
    (10, "Which social or reform movement contributed to a peaceful transition to democracy?", ["Solidarity"], "word", "Persons"),

    # Cat 2: Dates / Years / Numbers
    (11, "In which year did Duke Mieszko I adopt Western Christianity?", ["966", "966 AD"], "word", "Dates/Numbers"),
    (12, "In what year was the medieval kingdom formally reconstituted?", ["1025"], "word", "Dates/Numbers"),
    (13, "In which year did Casimir III the Great die?", ["1370"], "word", "Dates/Numbers"),
    (14, "In what year was the Polish–Lithuanian Commonwealth established?", ["1569"], "word", "Dates/Numbers"),
    (15, "What was the date of Europe's first modern constitution mentioned in the text?", ["3 May 1791", "May 1791", "1791"], "word", "Dates/Numbers"),
    (16, "In what year did the existence of the Commonwealth end?", ["1795"], "word", "Dates/Numbers"),
    (17, "In which year was the Second Polish Republic established?", ["1918"], "word", "Dates/Numbers"),
    (18, "Until which year was there no truly independent Polish state after 1795?", ["1918"], "word", "Dates/Numbers"),
    (19, "Between which years did millions of Polish citizens perish under Nazi occupation?", ["1939 and 1945", "1939", "1945"], "word", "Dates/Numbers"),
    (20, "In what year was the modern Third Polish Republic founded?", ["1989"], "word", "Dates/Numbers"),

    # Cat 3: Entities / Dynasties / Concepts
    (21, "Which ruling dynasty emerged in the 10th century AD?", ["Piasts", "The Piasts"], "word", "Entities/Concepts"),
    (22, "Which dynasty brought close ties with Lithuania between the 14th and 16th centuries?", ["Jagiellonian", "Jagiellonian dynasty"], "word", "Entities/Concepts"),
    (23, "Which tribe gave Poland its name?", ["Lechitic Western Polans", "Western Polans", "Polans"], "word", "Entities/Concepts"),
    (24, "What does the name of the Western Polans denote?", ["people living in open fields"], "word", "Entities/Concepts"),
    (25, "In which geographic plain does Poland lie?", ["North-Central European Plain"], "word", "Entities/Concepts"),
    (26, "What union was established in 1569?", ["Polish–Lithuanian Commonwealth", "Commonwealth"], "word", "Entities/Concepts"),
    (27, "What unique political system matured in the Commonwealth?", ["noble democracy"], "word", "Entities/Concepts"),
    (28, "Which three powers partitioned Poland in 1795?", ["Russian Empire", "Kingdom of Prussia", "Habsburg monarchy"], "word", "Entities/Concepts"),
    (29, "Which two powers invaded Poland in 1939?", ["Nazi Germany and the Soviet Union", "Nazi Germany", "Soviet Union"], "word", "Entities/Concepts"),
    (30, "What was the official name of the communist state known from 1952?", ["Polish People's Republic"], "word", "Entities/Concepts"),

    # Cat 4: Boolean (Yes / No)
    (31, "Did Duke Mieszko I adopt Western Christianity in 966 AD? Answer strictly Yes or No.", ["Yes"], "word", "Boolean"),
    (32, "Was Casimir III the Great the last monarch of the Piast dynasty? Answer strictly Yes or No.", ["Yes"], "word", "Boolean"),
    (33, "Did the Polish–Lithuanian Commonwealth establish an absolute monarchy? Answer strictly Yes or No.", ["No"], "word", "Boolean"),
    (34, "Was the Second Polish Republic established in 1918? Answer strictly Yes or No.", ["Yes"], "word", "Boolean"),
    (35, "Did an independent Polish state exist continuously throughout the 19th century? Answer strictly Yes or No.", ["No"], "word", "Boolean"),
    (36, "Was Europe's first constitution enacted on 3 May 1791? Answer strictly Yes or No.", ["Yes"], "word", "Boolean"),
    (37, "Did Nazi Germany and the Soviet Union invade Poland in 1939? Answer strictly Yes or No.", ["Yes"], "word", "Boolean"),
    (38, "Was the Third Polish Republic founded in 1945? Answer strictly Yes or No.", ["No"], "word", "Boolean"),
    (39, "Is this text mainly about the history of Poland? Answer strictly Yes or No.", ["Yes"], "word", "Boolean"),
    (40, "Is this text primarily focused on the Ottoman Empire? Answer strictly Yes or No.", ["No"], "word", "Boolean"),

    # Cat 5: Refusal Gate (Unmentioned facts)
    (41, "Who was the king during the Battle of Grunwald in 1410?", ["No information provided in the given context."], "word", "Refusal"),
    (42, "In which year did Poland join the European Union?", ["No information provided in the given context."], "word", "Refusal"),
    (43, "What was the role of Napoleon Bonaparte in the Duchy of Warsaw?", ["No information provided in the given context."], "word", "Refusal"),
    (44, "How many casualties were suffered during the Warsaw Uprising?", ["No information provided in the given context."], "word", "Refusal"),
    (45, "Who was elected as the first President of Poland in 1990?", ["No information provided in the given context."], "word", "Refusal"),

    # Cat 6: Typos & Robustness
    (46, "Who was Miesko I?", ["Duke Mieszko I", "creator of Polish statehood", "Mieszko"], "word", "Typo/Robustness"),
    (47, "In what year did Boleslaw the Brave reconstitute the kingdom?", ["1025"], "word", "Typo/Robustness"),
    (48, "Which dynasty followed the Piast kings in the 14th century?", ["Jagiellonian", "Jagiellonian dynasty"], "word", "Typo/Robustness"),
    (49, "Did Kasimir the Great die in 1370? Answer strictly Yes or No.", ["Yes"], "word", "Typo/Robustness"),

    # Cat 7: Paragraph Synthesis
    (50, "Summarize the major historical transformations of Poland described in the article.", ["Poland", "century", "state"], "paragraph", "Synthesis"),
]


def main():
    print("=" * 80)
    print("  EVALUATION OF 50 QUESTIONS ON POLISH HISTORY (BIO-CASCADE V13.0)")
    print("=" * 80)

    tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
    model_path = BASE_DIR / "models" / "bio_cascade_generative_live.pkl"

    model = BioCascadeGenerativeDecoder(str(tokenizer_path))
    model.load_checkpoint(model_path)
    model.eval()

    results = []
    latencies = []

    for qid, q_text, expected_list, mode, cat in QUESTIONS:
        t0 = time.perf_counter()
        try:
            res = model.generate_response(CONTEXT, q_text, mode=mode, max_new_tokens=60)
            out_text = res["generated_text"].strip()
        except Exception as e:
            out_text = f"CRASH: {type(e).__name__}: {e}"
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        is_pass = False
        if cat == "Refusal":
            is_pass = ("no information provided" in out_text.lower() or "brak informacji" in out_text.lower())
        elif cat == "Boolean":
            is_pass = (out_text.lower() == expected_list[0].lower())
        elif cat == "Synthesis":
            is_pass = len(out_text.split()) >= 15 and any(k.lower() in out_text.lower() for k in expected_list)
        else:
            is_pass = any(exp.lower() in out_text.lower() or out_text.lower() in exp.lower() for exp in expected_list)

        results.append({
            "id": qid,
            "category": cat,
            "question": q_text,
            "expected": expected_list,
            "output": out_text,
            "latency_ms": dt,
            "passed": is_pass,
            "mode": mode,
        })

        status_str = "PASS" if is_pass else "FAIL"
        print(f"[{status_str}] Q{qid:02d} ({dt:5.1f} ms) [{cat:<16}] Q: {q_text[:45]}...")
        print(f"       Out: '{out_text}' | Expected: {expected_list}")

    pass_count = sum(1 for r in results if r["passed"])
    total_q = len(results)
    acc = (pass_count / total_q) * 100.0
    avg_lat = sum(latencies) / len(latencies)

    print("\n" + "=" * 80)
    print(f"  TOTAL ACCURACY: {pass_count}/{total_q} ({acc:.1f}%)")
    print(f"  LATENCY: Min: {min(latencies):.1f} ms | Avg: {avg_lat:.1f} ms | Max: {max(latencies):.1f} ms")
    print("=" * 80)

    sys.exit(0 if pass_count == total_q else 1)


if __name__ == "__main__":
    main()
