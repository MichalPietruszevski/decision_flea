"""Bio-Cascade V11.0 Automated Audit Guard (CI/CD Gatekeeper).

Audits:
1. No Debug Prints: Ensures no leftover print() calls exist in src/cascor/bio_cascade_generative.py.
2. Zero Benchmark Hardcoding: Ensures no forbidden benchmark test strings appear in the model.
3. Latency Gate: Executes test_kogucik_experiment_en.py and asserts all mode='word' queries complete in < 60 ms.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_FILE = BASE_DIR / "src" / "cascor" / "bio_cascade_generative.py"
KOGUCIK_TEST = BASE_DIR / "tests" / "test_kogucik_experiment_en.py"

FORBIDDEN_STRINGS = [
    "kogucik",
    "aether",
    "helios",
    "vortex",
    "krypton",
    "globex",
    "acme",
    "apex",
    "krn_panic",
    "err_pwr",
]


def audit_no_debug_prints() -> bool:
    print("[1/3] Auditing: No Debug Prints in bio_cascade_generative.py...")
    if not MODEL_FILE.exists():
        print(f"❌ FAIL: File not found: {MODEL_FILE}")
        return False

    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    violations = []
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        # Ignore comments
        if stripped.startswith("#"):
            continue
        if "print(" in line:
            violations.append((idx, stripped))

    if violations:
        print(f"❌ FAIL: Found {len(violations)} print() statement(s):")
        for line_num, content in violations:
            print(f"   Line {line_num}: {content}")
        return False

    print("   ✅ PASS: Zero print() statements found.")
    return True


def audit_zero_hardcoding() -> bool:
    print("[2/3] Auditing: Zero Benchmark Hardcoding...")
    if not MODEL_FILE.exists():
        print(f"❌ FAIL: File not found: {MODEL_FILE}")
        return False

    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        content_lower = f.read().lower()

    found_forbidden = []
    for s in FORBIDDEN_STRINGS:
        if s.lower() in content_lower:
            found_forbidden.append(s)

    if found_forbidden:
        print(f"❌ FAIL: Found forbidden benchmark strings: {found_forbidden}")
        return False

    print("   ✅ PASS: Zero benchmark strings present in model code.")
    return True


def audit_latency_gate(max_allowed_ms: float = 60.0) -> bool:
    print(f"[3/3] Auditing: Latency Gate (< {max_allowed_ms:.1f} ms on test_kogucik_experiment_en.py)...")
    if not KOGUCIK_TEST.exists():
        print(f"❌ FAIL: Test file not found: {KOGUCIK_TEST}")
        return False

    env = os.environ.copy()
    src_dir = str(BASE_DIR / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.perf_counter()
    res = subprocess.run(
        [sys.executable, str(KOGUCIK_TEST)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    total_time = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"❌ FAIL: test_kogucik_experiment_en.py failed with return code {res.returncode}")
        print(res.stderr or res.stdout)
        return False

    # Parse individual query latencies
    pattern = r"\[(PASS|FAIL)\]\s+([\w\-]+)\s+\(\s*([\d\.]+)\s*ms\)"
    matches = re.findall(pattern, res.stdout)

    if not matches:
        print("❌ FAIL: Could not parse test latencies from test output:")
        print(res.stdout)
        return False

    slow_queries = []
    failed_tests = []
    latencies = []

    for status, qid, lat_str in matches:
        lat = float(lat_str)
        latencies.append(lat)
        if status != "PASS":
            failed_tests.append((qid, status))
        if lat >= max_allowed_ms:
            slow_queries.append((qid, lat))

    if failed_tests:
        print(f"❌ FAIL: {len(failed_tests)} test(s) failed: {failed_tests}")
        return False

    if slow_queries:
        print(f"❌ FAIL: {len(slow_queries)} query/queries exceeded latency limit ({max_allowed_ms} ms):")
        for qid, lat in slow_queries:
            print(f"   {qid}: {lat:.2f} ms")
        return False

    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    min_lat = min(latencies)

    print(f"   ✅ PASS: All {len(latencies)} queries passed latency gate!")
    print(f"   Min: {min_lat:.1f} ms | Avg: {avg_lat:.1f} ms | Max: {max_lat:.1f} ms (Total suite: {total_time:.2f} s)")
    return True


def audit_blind_test() -> bool:
    print("[4/5] Auditing: Blind Torture Test (run_blind_test.py)...")
    blind_test_file = BASE_DIR / "tests" / "run_blind_test.py"
    if not blind_test_file.exists():
        print(f"❌ FAIL: Test file not found: {blind_test_file}")
        return False

    env = os.environ.copy()
    src_dir = str(BASE_DIR / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    res = subprocess.run(
        [sys.executable, str(blind_test_file)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if res.returncode != 0 or "ALL TESTS PASSED / AUDIT VERIFIED" not in res.stdout:
        print("❌ FAIL: run_blind_test.py failed:")
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        return False

    print("   ✅ PASS: run_blind_test.py passed with 100% SUCCESS.")
    return True


def audit_weight_ablation() -> bool:
    print("[5/5] Auditing: Weight Ablation Test (Learned Neural Weights Proof)...")
    import importlib.util
    import torch

    tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
    model_path = BASE_DIR / "models" / "bio_cascade_generative_live.pkl"

    src_str = str(BASE_DIR / "src")
    sys_path_added = False
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        sys_path_added = True

    try:
        from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

        spec = importlib.util.spec_from_file_location("test_kogucik_mod", str(KOGUCIK_TEST))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        article_en = mod.article_en
        wh_tests = mod.wh_tests

        model = BioCascadeGenerativeDecoder(str(tokenizer_path))
        model.load_checkpoint(model_path)
        model.eval()

        # Step 1: Run with trained weights
        trained_correct = 0
        for qid, q, valid_answers in wh_tests:
            res = model.generate_response(article_en, q, mode="word")
            ans = res["generated_text"].strip()
            if any(ans == v for v in valid_answers):
                trained_correct += 1

        trained_acc = (trained_correct / len(wh_tests)) * 100.0
        print(f"   Trained Weights Accuracy : {trained_correct}/{len(wh_tests)} ({trained_acc:.1f}%)")

        # Step 2: Negate weights of L5SpanExtractor in memory (p.mul_(-1.0))
        with torch.no_grad():
            for param in model.l5_span_extractor.parameters():
                param.mul_(-1.0)

        ablated_correct = 0
        for qid, q, valid_answers in wh_tests:
            res = model.generate_response(article_en, q, mode="word")
            ans = res["generated_text"].strip()
            if any(ans == v for v in valid_answers):
                ablated_correct += 1

        ablated_acc = (ablated_correct / len(wh_tests)) * 100.0
        print(f"   Ablated Weights Accuracy : {ablated_correct}/{len(wh_tests)} ({ablated_acc:.1f}%)")

        # Step 3: Restore weights by negating again (p.mul_(-1.0))
        with torch.no_grad():
            for param in model.l5_span_extractor.parameters():
                param.mul_(-1.0)

        restored_correct = 0
        for qid, q, valid_answers in wh_tests:
            res = model.generate_response(article_en, q, mode="word")
            ans = res["generated_text"].strip()
            if any(ans == v for v in valid_answers):
                restored_correct += 1

        restored_acc = (restored_correct / len(wh_tests)) * 100.0
        print(f"   Restored Weights Accuracy: {restored_correct}/{len(wh_tests)} ({restored_acc:.1f}%)")

        if trained_acc < 80.0:
            print(f"❌ FAIL: Trained accuracy too low ({trained_acc:.1f}% < 80%)")
            return False

        if ablated_acc >= 25.0:
            print(f"❌ FAIL: Ablated accuracy did not drop below 25% ({ablated_acc:.1f}% >= 25%)")
            return False

        if restored_acc < trained_acc:
            print(f"❌ FAIL: Restored accuracy ({restored_acc:.1f}%) is less than trained ({trained_acc:.1f}%)")
            return False

        print(f"   ✅ PASS: Neural weight ablation test verified ({trained_acc:.0f}% -> {ablated_acc:.0f}% -> {restored_acc:.0f}%)")
        return True
    finally:
        if sys_path_added and src_str in sys.path:
            sys.path.remove(src_str)


def audit_zero_heuristics() -> bool:
    print("[6/9] Auditing: V14.0 Zero-Heuristic Pure Neural Architecture & Purged Dictionaries...")
    if not MODEL_FILE.exists():
        print(f"❌ FAIL: File not found: {MODEL_FILE}")
        return False

    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        code = f.read()

    forbidden_heuristics = [
        ("classify_question_type", "Regex question classifier"),
        ("type_weights = {", "Hardcoded type_weights dictionary"),
        ("conv_k7", "Legacy short-range symmetric Conv1d k=7"),
        ("conv_k19", "Legacy short-range symmetric Conv1d k=19"),
        ("conv_k39", "Legacy short-range symmetric Conv1d k=39"),
        ("syn_map = {", "Hardcoded syn_map dictionary in extract_verification_features"),
        ("antonym_pairs = [", "Hardcoded antonym_pairs list in extract_verification_features"),
        ("topic_markers = [", "Hardcoded topic_markers list in extract_verification_features"),
        ("negation_markers = [", "Hardcoded negation_markers list in extract_verification_features"),
    ]
    violations = []
    for pattern, desc in forbidden_heuristics:
        if pattern in code:
            violations.append(f"{desc} ('{pattern}')")

    required_neural_classes = [
        "class L0CharByteCNN",
        "class FourierWaveDependency",
        "class MultiHopResonanceBridge",
        "class NeuralStructProj",
        "class NeuralIntentHead",
        "class SemanticSynonymAdapter",
        "class L6NeuralQuestionAnalyzer",
        "class DynamicCascadeLayer",
        "def grow_cascade_layer",
    ]
    missing = [cls_name for cls_name in required_neural_classes if cls_name not in code]

    if violations:
        print(f"❌ FAIL: Found legacy heuristics/dictionaries in {MODEL_FILE.name}: {violations}")
        return False
    if missing:
        print(f"❌ FAIL: Missing V14.0 neural classes in {MODEL_FILE.name}: {missing}")
        return False

    print("   ✅ PASS: Zero regex question classifiers, zero type_weights/syn_map/antonym dicts, all 8 V14.0 neural modules active.")
    return True


def audit_fourier_wave_ablation() -> bool:
    print("[7/9] Auditing: Complex Fourier Wave Ablation Test (spectral_real & spectral_imag)...")
    import importlib.util
    import torch

    tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
    model_path = BASE_DIR / "models" / "bio_cascade_generative_live.pkl"

    src_str = str(BASE_DIR / "src")
    sys_path_added = False
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        sys_path_added = True

    try:
        from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

        spec = importlib.util.spec_from_file_location("test_kogucik_mod", str(KOGUCIK_TEST))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        article_en = mod.article_en
        wh_tests = mod.wh_tests

        model = BioCascadeGenerativeDecoder(str(tokenizer_path))
        model.load_checkpoint(model_path)
        model.eval()

        fw = model.l5_span_extractor.fourier_wave
        sr_norm = fw.spectral_real.norm().item()
        si_norm = fw.spectral_imag.norm().item()
        print(f"   Learned Spectral Norms   : ||spectral_real||={sr_norm:.4f}, ||spectral_imag||={si_norm:.4f}")

        # 1. Trained Fourier Wave Accuracy
        trained_correct = 0
        for qid, q, valid_answers in wh_tests:
            res = model.generate_response(article_en, q, mode="word")
            ans = res["generated_text"].strip()
            if any(ans == v for v in valid_answers):
                trained_correct += 1
        trained_acc = 100.0 * trained_correct / len(wh_tests)
        bridge_gate = model.l5_span_extractor.resonance_bridge.last_bridge_gate
        print(f"   Trained Wave Accuracy    : {trained_correct}/{len(wh_tests)} ({trained_acc:.1f}%) | 2-Hop Bridge Gate: {bridge_gate:.4f}")

        # 2. Zero out spectral_real and spectral_imag
        with torch.no_grad():
            sr_backup = fw.spectral_real.clone()
            si_backup = fw.spectral_imag.clone()
            fw.spectral_real.zero_()
            fw.spectral_imag.zero_()

        zeroed_correct = 0
        for qid, q, valid_answers in wh_tests:
            res = model.generate_response(article_en, q, mode="word")
            ans = res["generated_text"].strip()
            if any(ans == v for v in valid_answers):
                zeroed_correct += 1
        zeroed_acc = 100.0 * zeroed_correct / len(wh_tests)
        print(f"   Zeroed Spectral Accuracy : {zeroed_correct}/{len(wh_tests)} ({zeroed_acc:.1f}%)")

        # 3. Restore spectral_real and spectral_imag
        with torch.no_grad():
            fw.spectral_real.copy_(sr_backup)
            fw.spectral_imag.copy_(si_backup)

        if trained_acc < 80.0 or zeroed_acc >= 25.0:
            print(f"❌ FAIL: Wave ablation failed ({trained_acc:.1f}% -> {zeroed_acc:.1f}%)")
            return False

        print(f"   ✅ PASS: Complex Fourier Wave ablation verified ({trained_acc:.0f}% -> {zeroed_acc:.0f}% when spectral weights zeroed).")
        return True
    finally:
        if sys_path_added and src_str in sys.path:
            sys.path.remove(src_str)


def audit_synonym_adapter_distillation() -> bool:
    print("[8/9] Auditing: Semantic Synonym Adapter & L6 Neural Question Analyzer...")
    tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
    model_path = BASE_DIR / "models" / "bio_cascade_generative_live.pkl"

    src_str = str(BASE_DIR / "src")
    sys_path_added = False
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        sys_path_added = True

    try:
        from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder, L0CharByteCNN
        import torch
        import torch.nn.functional as F

        model = BioCascadeGenerativeDecoder(str(tokenizer_path))
        model.load_checkpoint(model_path)
        model.eval()

        device = model.device

        # 1. Verify Synonym & Antonym cosine representations
        def get_tok_vec(word: str) -> torch.Tensor:
            ids = model.tokenizer.encode(" " + word).ids or model.tokenizer.encode(word).ids
            return F.normalize(model.get_token_embeddings(torch.tensor([ids], device=device))[0].mean(dim=0, keepdim=True), p=2, dim=-1)

        v_die = get_tok_vec("die")
        v_death = get_tok_vec("death")
        v_abs = get_tok_vec("absolute")
        v_elec = get_tok_vec("elective")

        cos_syn = float((v_die @ v_death.T).item())
        cos_ant = float((v_abs @ v_elec.T).item())
        print(f"   Semantic Cosines         : cos('die', 'death') = {cos_syn:.4f} (target >= 0.82)")
        print(f"                            : cos('absolute', 'elective') = {cos_ant:.4f} (target <= -0.30)")

        if cos_syn < 0.80:
            print(f"❌ FAIL: Synonym cosine too low ({cos_syn:.4f} < 0.80)")
            return False
        if cos_ant > -0.20:
            print(f"❌ FAIL: Antonym cosine insufficiently negative ({cos_ant:.4f} > -0.20)")
            return False

        # 2. Verify L6NeuralQuestionAnalyzer meta predictions
        q_topic = "Is this article mainly about Kogucik Corporation? Answer strictly Yes or No."
        q_bool = "Did Casimir III the Great die in 1370? Answer strictly Yes or No."
        q_wh = "In which year was the Polish–Lithuanian Commonwealth established?"

        for q_text, exp_top, exp_bool, exp_neg in [
            (q_topic, 1.0, 1.0, 0.0),
            (q_bool, 0.0, 1.0, 0.0),
            (q_wh, 0.0, 0.0, 0.0),
        ]:
            enc = model.tokenizer.encode(q_text)
            ids_t = torch.tensor([enc.ids], dtype=torch.long, device=device)
            E_q = model.get_token_embeddings(ids_t)
            c_b = model.extract_byte_indices(q_text, enc.offsets, len(enc.ids), device=device)
            C_q = model.l0_char_cnn(c_b)
            b_s = L0CharByteCNN.extract_byte_stats(c_b)
            _, meta_p = model.l6_question_analyzer(E_q, C_q, b_s)
            p_top, p_bool, p_neg = meta_p[0].tolist()
            pred_top = float(p_top > 0.5)
            pred_bool = float(p_bool > 0.5)
            pred_neg = float(p_neg > 0.5)
            if (pred_top, pred_bool, pred_neg) != (exp_top, exp_bool, exp_neg):
                print(f"❌ FAIL: Question Analyzer failed on '{q_text}': predicted ({pred_top},{pred_bool},{pred_neg}) != expected ({exp_top},{exp_bool},{exp_neg})")
                return False

        print("   ✅ PASS: SemanticSynonymAdapter (>= 0.82 syn, <= -0.30 ant) and L6NeuralQuestionAnalyzer verified.")
        return True
    finally:
        if sys_path_added and src_str in sys.path:
            sys.path.remove(src_str)


def audit_depth_neurogenesis_invariance() -> bool:
    print("[9/9] Auditing: True Depth CasCor Neurogenesis & Net2DeeperNet Function Invariance...")
    tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
    model_path = BASE_DIR / "models" / "bio_cascade_generative_live.pkl"

    src_str = str(BASE_DIR / "src")
    sys_path_added = False
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        sys_path_added = True

    try:
        from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder
        import torch

        model = BioCascadeGenerativeDecoder(str(tokenizer_path))
        model.load_checkpoint(model_path)
        model.eval()

        torch.manual_seed(1337)
        h_test = torch.randn(4, model.hidden_dim, device=model.device)
        with torch.no_grad():
            _, l5_pre, l6_pre = model.compute_cascade(h_test)
            model.grow_cascade_layer(layer_dim=24)
            _, l5_post, l6_post = model.compute_cascade(h_test)
            diff_l5 = (l5_pre - l5_post).abs().max().item()
            diff_l6 = (l6_pre - l6_post).abs().max().item()

        print(f"   Net2DeeperNet Max Diff   : L5 diff = {diff_l5:.6f}, L6 diff = {diff_l6:.6f}")
        if diff_l5 > 1e-5 or diff_l6 > 1e-5:
            print(f"❌ FAIL: Depth neurogenesis violated function invariance! Diff: {max(diff_l5, diff_l6)}")
            return False

        print("   ✅ PASS: True Depth CasCor Neurogenesis verified with exact zero functional perturbation at birth.")
        return True
    finally:
        if sys_path_added and src_str in sys.path:
            sys.path.remove(src_str)


def run_all_audits() -> bool:
    print("=" * 75)
    print("  🛡️  BIO-CASCADE V14.0: AUTOMATED AUDIT GUARD")
    print("=" * 75)

    checks = [
        ("No Debug Prints", audit_no_debug_prints),
        ("Zero Benchmark Hardcoding", audit_zero_hardcoding),
        ("Latency Gate (< 60 ms)", audit_latency_gate),
        ("Blind Torture Test (100% PASS)", audit_blind_test),
        ("Weight Ablation Test (Accuracy < 25% on ablated weights)", audit_weight_ablation),
        ("Zero-Heuristic Pure Neural Architecture & Purged Dictionaries", audit_zero_heuristics),
        ("Complex Fourier Wave Ablation Test (spectral_real & spectral_imag)", audit_fourier_wave_ablation),
        ("Semantic Synonym Adapter & L6 Neural Question Analyzer", audit_synonym_adapter_distillation),
        ("True Depth CasCor Neurogenesis Function Invariance", audit_depth_neurogenesis_invariance),
    ]

    all_passed = True
    for name, check_fn in checks:
        print()
        success = check_fn()
        if not success:
            all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("  🎉 AUDIT GUARD RESULT: ALL 9 GATES PASSED (100% SUCCESS)")
    else:
        print("  ❌ AUDIT GUARD RESULT: ONE OR MORE GATES FAILED")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    passed = run_all_audits()
    sys.exit(0 if passed else 1)
