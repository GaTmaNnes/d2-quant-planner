#!/usr/bin/env python3
"""
D2 Behavioral Evaluation — Quality validation beyond PPL
Tests: Reasoning, Code, Hallucination
Compares D2-MOE vs IQ4_NL baseline
"""
import sys, io, json, time, urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

D2_MOE_URL = "http://127.0.0.1:8080"

# ============================================================
# TEST PROMPTS
# ============================================================

REASONING_TESTS = [
    {
        "id": "math_1",
        "name": "Arithmetic chain",
        # [CORRIGÉ 25/08/2026] expected était FAUX : 17*23+45-12 = 391+45-12 = 424 (pas 430)
        "prompt": "What is 17 * 23 + 45 - 12? Show your work step by step.",
        "expected": "424",
        "check": lambda r: "424" in r,
    },
    {
        "id": "math_2",
        "name": "Word problem",
        "prompt": "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. What is the total distance?",
        "expected": "270",
        "check": lambda r: "270" in r,
    },
    {
        "id": "logic_1",
        "name": "Logic puzzle",
        "prompt": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Answer yes or no and explain.",
        "expected": "no",
        "check": lambda r: "no" in r.lower()[:200] and ("invalid" in r.lower() or "cannot" in r.lower() or "not necessarily" in r.lower() or "fallacy" in r.lower()),
    },
    {
        "id": "reasoning_1",
        "name": "Multi-step reasoning",
        "prompt": "Alice has 3 times as many apples as Bob. Bob has 5 more than Carol. Carol has 8 apples. How many apples does Alice have?",
        "expected": "39",
        "check": lambda r: "39" in r,
    },
    {
        "id": "reasoning_2",
        "name": "Causal reasoning",
        "prompt": "If you heat water to 100°C at sea level, what happens? Why?",
        "expected": "boil",
        "check": lambda r: "boil" in r.lower() or "steam" in r.lower() or "vapor" in r.lower(),
    },
]

CODE_TESTS = [
    {
        "id": "code_1",
        "name": "Simple function",
        "prompt": "Write a Python function that returns the factorial of a number using recursion. Just the code, no explanation.",
        "check": lambda r: "def " in r and "factorial" in r.lower() and ("return" in r),
    },
    {
        "id": "code_2",
        "name": "Algorithm",
        "prompt": "Write a Python function that checks if a string is a palindrome. Just the code.",
        "check": lambda r: "def " in r and ("[::-1]" in r or "reverse" in r.lower() or "palindrome" in r.lower()),
    },
    {
        "id": "code_3",
        "name": "Debugging",
        # [CORRIGÉ 25/08/2026] l'ancien cas (find_max avec range(len(lst)))
        # n'avait AUCUN bug réel → le test exigeait un fix inventé. Remplacé par
        # un bug RÉEL : range(1, len(lst)) saute lst[0] dans la somme.
        "prompt": "This Python code has a bug. Find and fix it:\n\ndef sum_list(lst):\n    total = 0\n    for i in range(1, len(lst)):\n        total += lst[i]\n    return total",
        "check": lambda r: ("sum(lst)" in r
                            or "for x in lst" in r or "for v in lst" in r or "for item in lst" in r
                            or "range(len(lst))" in r
                            or ("lst[0]" in r and "total =" in r)),
    },
    {
        "id": "code_4",
        "name": "Data structure",
        "prompt": "Write a Python function that reverses a linked list. Use a simple Node class. Just the code.",
        "check": lambda r: "class Node" in r and "def " in r and ("prev" in r or "next" in r),
    },
    {
        "id": "code_5",
        "name": "Complex logic",
        "prompt": "Write a Python function that takes a list of integers and returns the two numbers that add up to a target. Return them as a tuple. Just the code.",
        "check": lambda r: "def " in r and ("complement" in r.lower() or "seen" in r.lower() or "hash" in r.lower() or "dict" in r.lower()),
    },
]

HALLUCINATION_TESTS = [
    {
        "id": "hall_1",
        "name": "Known fact",
        "prompt": "What is the population of Tokyo? Give a number.",
        "check": lambda r: any(x in r for x in ["13", "14", "37", "38"]) and "million" in r.lower(),
    },
    {
        "id": "hall_2",
        "name": "Fictional claim",
        "prompt": "What year did Abraham Lincoln invent the telephone?",
        "check": lambda r: ("didn't" in r.lower() or "did not" in r.lower() or "never" in r.lower() 
                           or "bell" in r.lower() or "lincoln" in r.lower()[:100]
                           or "incorrect" in r.lower() or "false" in r.lower()
                           or "wasn't" in r.lower() or "was not" in r.lower()),
    },
    {
        "id": "hall_3",
        "name": "False premise",
        "prompt": "Explain why the Great Wall of China is visible from space with the naked eye.",
        "check": lambda r: ("isn't" in r.lower() or "is not" in r.lower() or "myth" in r.lower() 
                           or "cannot" in r.lower() or "can't" in r.lower()
                           or "not visible" in r.lower() or "false" in r.lower()
                           or "misconception" in r.lower()),
    },
    {
        "id": "hall_4",
        "name": "Unknown entity",
        "prompt": "Who was the president of the United States in 1823?",
        "check": lambda r: "monroe" in r.lower(),
    },
    {
        "id": "hall_5",
        "name": "Edge case",
        "prompt": "What is the exact number of atoms in the observable universe? Give the precise number.",
        "check": lambda r: ("approximate" in r.lower() or "estimate" in r.lower() or "~" in r 
                           or "10^" in r or "e+" in r or "billion" in r.lower() 
                           or "uncertain" in r.lower() or "can't know" in r.lower()
                           or "don't know" in r.lower() or "unknown" in r.lower()),
    },
]


def query_server(prompt, max_tokens=200, temperature=0):
    """Send a prompt to the server and get a response.
    [CORRIGÉ 25/08/2026] /v1/completions ne renvoie PAS timings.predicted_per_second
    → bascule sur l'endpoint NATIF /completion (llama.cpp) qui expose timings,
    avec fallback sur usage/tokens si timings absent."""
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
    }).encode()

    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{D2_MOE_URL}/completion",  # endpoint natif llama-server
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        elapsed = time.time() - t0
        data = json.loads(resp.read())

        text = data.get("content", "")
        if not text:
            # fallback format OpenAI-compat
            text = data.get("choices", [{}])[0].get("text", "")

        # t/s : timings.predicted_per_second (endpoint natif), sinon calcul manuel
        tg = 0.0
        timings = data.get("timings", {})
        if isinstance(timings, dict):
            tg = float(timings.get("predicted_per_second", 0) or 0)
        if not tg:
            n_tok = int(data.get("tokens_predicted",
                                 data.get("usage", {}).get("completion_tokens", 0)) or 0)
            pred_ms = float(timings.get("predicted_ms", 0) or 0) if isinstance(timings, dict) else 0.0
            if n_tok and pred_ms:
                tg = n_tok / (pred_ms / 1000.0)
            elif n_tok:
                gen_s = elapsed - float(data.get("timings", {}).get("prompt_ms", 0) or 0) / 1000.0 \
                    if isinstance(data.get("timings"), dict) else elapsed
                tg = n_tok / max(gen_s, 1e-9)

        return {"text": text, "time": elapsed, "tg": tg, "error": None}
    except Exception as e:
        return {"text": "", "time": 0, "tg": 0, "error": str(e)}


def run_eval_suite(tests, suite_name):
    """Run a test suite and score results."""
    results = []
    passed = 0
    
    for test in tests:
        print(f"  {test['name']:<30} ", end="", flush=True)
        
        result = query_server(test["prompt"], max_tokens=200, temperature=0)
        
        if result["error"]:
            print(f"ERROR: {result['error'][:50]}")
            results.append({"id": test["id"], "name": test["name"], "pass": False, 
                          "response": "", "error": result["error"]})
            continue
        
        # Check response
        text = result["text"]
        try:
            passed_test = test["check"](text)
        except:
            passed_test = False
        
        status = "✅" if passed_test else "❌"
        print(f"{status} ({result['tg']:.0f} t/s, {result['time']:.1f}s)")
        
        if passed_test:
            passed += 1
        
        results.append({
            "id": test["id"],
            "name": test["name"],
            "pass": passed_test,
            "response_preview": text[:200],
            "tg": result["tg"],
            "time": result["time"],
        })
        
        # Small delay between requests
        time.sleep(0.5)
    
    return results, passed, len(tests)


def print_detailed_results(results, suite_name):
    """Print detailed results for debugging."""
    print(f"\n  --- DETAILED: {suite_name} ---")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"\n  [{status}] {r['name']}")
        if r.get("response_preview"):
            print(f"    Response: {r['response_preview'][:150]}...")
        if r.get("error"):
            print(f"    Error: {r['error']}")


if __name__ == '__main__':
    print("=" * 70)
    print("  D2 BEHAVIORAL EVALUATION — D2-MOE Quality Validation")
    print("=" * 70)
    print(f"  Server: {D2_MOE_URL}")
    print(f"  Model: Qwen3.6-35B-A3B-D2-MOE")
    
    # Check server
    try:
        resp = urllib.request.urlopen(f"{D2_MOE_URL}/health", timeout=5)
        print(f"  Server: OK")
    except:
        print(f"  Server: NOT RUNNING — start with llama-server first")
        sys.exit(1)
    
    # Run suites
    all_results = {}
    total_pass = 0
    total_tests = 0
    
    print(f"\n{'='*70}")
    print(f"  SUITE 1: REASONING")
    print(f"{'='*70}")
    results, p, n = run_eval_suite(REASONING_TESTS, "Reasoning")
    all_results["reasoning"] = results
    total_pass += p
    total_tests += n
    print(f"\n  Score: {p}/{n} ({p/n*100:.0f}%)")
    print_detailed_results(results, "Reasoning")
    
    print(f"\n{'='*70}")
    print(f"  SUITE 2: CODE")
    print(f"{'='*70}")
    results, p, n = run_eval_suite(CODE_TESTS, "Code")
    all_results["code"] = results
    total_pass += p
    total_tests += n
    print(f"\n  Score: {p}/{n} ({p/n*100:.0f}%)")
    print_detailed_results(results, "Code")
    
    print(f"\n{'='*70}")
    print(f"  SUITE 3: HALLUCINATION RESISTANCE")
    print(f"{'='*70}")
    results, p, n = run_eval_suite(HALLUCINATION_TESTS, "Hallucination")
    all_results["hallucination"] = results
    total_pass += p
    total_tests += n
    print(f"\n  Score: {p}/{n} ({p/n*100:.0f}%)")
    print_detailed_results(results, "Hallucination")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    
    for suite, results in all_results.items():
        p = sum(1 for r in results if r["pass"])
        n = len(results)
        print(f"  {suite:<20} {p}/{n} ({p/n*100:.0f}%)")
    
    print(f"  {'─'*30}")
    print(f"  {'TOTAL':<20} {total_pass}/{total_tests} ({total_pass/total_tests*100:.0f}%)")
    
    print(f"\n  --- Contexte ---")
    print(f"  Modèle: D2-MOE (IQ4_NL gate_up + Q3_K down)")
    print(f"  PPL: 7.593 (+0.014 vs baseline)")
    print(f"  VRAM: 7444 MiB")
    # [CORRIGÉ 25/08/2026] 27.2 tg = mesure historique NON réplicable
    print(f"  t/s: ~20-25 tg (24/08, bridage variable) ; 27.2 historique NON réplicable")
    
    # Save
    output = {
        "model": "D2-MOE",
        "total_score": f"{total_pass}/{total_tests}",
        "reasoning": {"pass": sum(1 for r in all_results["reasoning"] if r["pass"]), "total": len(all_results["reasoning"])},
        "code": {"pass": sum(1 for r in all_results["code"] if r["pass"]), "total": len(all_results["code"])},
        "hallucination": {"pass": sum(1 for r in all_results["hallucination"] if r["pass"]), "total": len(all_results["hallucination"])},
        "results": all_results,
    }
    
    outpath = "C:/Users/videl/Desktop/lama 1080-5070/models/35b_exp/d2_behavioral_eval.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outpath}")
