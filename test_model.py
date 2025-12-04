"""
Test pour voir EXACTEMENT ce que le modèle reçoit et attend.
"""
import sys
from pathlib import Path
import pandas as pd
from radon.raw import analyze as radon_raw_analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.preprocessing import helper_for_complexity_evaluation
    HAS_HELPER = True
except Exception:
    HAS_HELPER = False

# Code 1: PROPRE (tous les seuils en DESSOUS)
CODE_CLEAN = '''def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def main():
    x = 5
    y = 10
    print(add(x, y))
    print(multiply(x, y))

if __name__ == "__main__":
    main()
'''

# Code 2: COMPLEXE (au moins UN seuil ATTEINT)
CODE_COMPLEX = '''def ultra_complex(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
    """Code with massive parameters and imbrication."""
    results = {}
    for var1 in range(50):
        for var2 in range(50):
            for var3 in range(50):
                if a > 0:
                    if b > 0:
                        if c > 0:
                            if d > 0:
                                if e > 0:
                                    if f > 0:
                                        if g > 0:
                                            if h > 0:
                                                if i > 0:
                                                    if j > 0:
                                                        if k > 0:
                                                            if l > 0:
                                                                if m > 0:
                                                                    if n > 0:
                                                                        if o > 0:
                                                                            if p > 0:
                                                                                if q > 0:
                                                                                    if r > 0:
                                                                                        if s > 0:
                                                                                            if t > 0:
                                                                                                if u > 0:
                                                                                                    results[f"key_{var1}_{var2}_{var3}"] = a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y+z
    return results
'''

def extract_and_display(code_str, code_name):
    print(f"\n{'='*80}")
    print(f"TEST: {code_name}")
    print(f"{'='*80}\n")
    
    # Extraction
    raw = radon_raw_analyze(code_str)
    loc = raw.loc
    sloc = raw.sloc
    comments = raw.comments
    blank = raw.blank
    lloc = raw.lloc
    
    try:
        hal = h_visit(code_str).total
    except Exception:
        hal = None
    
    if hal is not None:
        n = getattr(hal, "length", 0)
        v = getattr(hal, "volume", 0)
        l = (v / n) if n > 0 else 0
        d = getattr(hal, "difficulty", 0)
        effort = getattr(hal, "effort", 0)
        time_metric = getattr(hal, "time", 0)
        uniq_op = getattr(hal, "h1", 0)
        uniq_opnd = getattr(hal, "h2", 0)
        total_op = getattr(hal, "N1", 0)
        total_opnd = getattr(hal, "N2", 0)
    else:
        n = v = l = d = effort = time_metric = uniq_op = uniq_opnd = total_op = total_opnd = 0
    
    try:
        cc_list = cc_visit(code_str)
        cyclomatic = max((c.complexity for c in cc_list), default=0)
    except Exception:
        cyclomatic = 0
    
    # Evaluation
    if HAS_HELPER:
        try:
            evaluation = helper_for_complexity_evaluation(n, v, d, effort, time_metric)
        except Exception:
            evaluation = int(((n >= 300) or (v >= 1000) or (d >= 50) or (effort >= 500000) or (time_metric >= 5000)))
    else:
        evaluation = int(((n >= 300) or (v >= 1000) or (d >= 50) or (effort >= 500000) or (time_metric >= 5000)))
    
    # Affiche les seuils
    print("📊 SEUILS D'ÉVALUATION:")
    print(f"  n (halstead_length) >= 300?     {n:>10.0f} >= 300    = {n >= 300}")
    print(f"  v (halstead_volume) >= 1000?    {v:>10.0f} >= 1000   = {v >= 1000}")
    print(f"  d (difficulty) >= 50?           {d:>10.2f} >= 50     = {d >= 50}")
    print(f"  effort >= 500000?               {effort:>10.0f} >= 500000 = {effort >= 500000}")
    print(f"  time >= 5000?                   {time_metric:>10.0f} >= 5000   = {time_metric >= 5000}")
    
    print(f"\n⚡ EVALUATION: {evaluation}")
    print(f"   → {('🐛 BUGGY (au moins un seuil atteint)' if evaluation == 1 else '✅ PROPRE (aucun seuil atteint)')}")
    
    # Les 18 features
    features = [
        loc, cyclomatic, n, v, l, d, (v / d if d > 0 else 0), effort, time_metric,
        sloc, comments, blank, lloc,
        uniq_op, uniq_opnd, total_op, total_opnd, evaluation
    ]
    
    feat_names = [
        'LOC', 'cyclomatic', 'n', 'v', 'l', 'd', 'i', 'e', 't',
        'sloc', 'comments', 'blank', 'lloc',
        'uniq_op', 'uniq_opnd', 'total_op', 'total_opnd', 'evaluation'
    ]
    
    print(f"\n📋 18 FEATURES ENVOYÉES AU MODÈLE:")
    for name, value in zip(feat_names, features):
        print(f"  {name:>18} = {value:>15.2f}")
    
    # DataFrame
    df = pd.DataFrame([dict(zip(feat_names, features))])
    print(f"\n✅ DataFrame (format model input):")
    print(df.to_string())
    
    return evaluation

if __name__ == "__main__":
    eval_clean = extract_and_display(CODE_CLEAN, "✅ CODE PROPRE")
    eval_complex = extract_and_display(CODE_COMPLEX, "🐛 CODE COMPLEXE")
    
    print(f"\n{'='*80}")
    print("RÉSUMÉ ATTENDU:")
    print(f"{'='*80}")
    print(f"Code propre:    evaluation={eval_clean} → Modèle doit dire ✅ PROPRE")
    print(f"Code complexe:  evaluation={eval_complex} → Modèle doit dire 🐛 BUGGY")