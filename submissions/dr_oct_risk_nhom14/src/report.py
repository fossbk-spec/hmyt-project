"""Compare your runs with the numbers REPORTED in the two papers.
  python report.py runs/fv_3d runs/fv_vit runs/olives_r18"""
import json, sys
REF = {  # copied from the papers (test, mean +- std)
 "FairVision DR, OCT 3D ResNet baseline : Overall AUC 92.22 +- 0.53 ; ES-AUC(race) 81.97": 92.22,
 "FairVision DR, OCT 3D ResNet + FIS     : Overall AUC 93.37 +- 0.25 ; ES-AUC(race) 85.78": 93.37,
 "FairVision DR, SLO ViT-B (2D) baseline : Overall AUC 85.22 +- 0.21": 85.22,
 "OLIVES DR/DME, OCT ResNet-18          : balanced acc 70.15 +- 4.69": 70.15,
 "OLIVES DR/DME, OCT + biomarker (joint): balanced acc 82.33 +- 3.59": 82.33}
print("Paper reference:"); [print("  ", k) for k in REF]
print("\nYour runs:")
for d in sys.argv[1:]:
    try: s = json.load(open(f"{d}/summary.json"))
    except Exception as e: print(d, "no summary", e); continue
    auc, bacc = s["test_auc"], s["test_bacc_at_val_thr"]
    print(f"{d:30s} test AUC {100*auc[0]:.2f} +- {100*auc[1]:.2f} | bAcc@valthr {100*bacc[0]:.2f} +- {100*bacc[1]:.2f}")
    for k, v in s.items():
        if k.startswith("fair_") and "es_auc" in v:
            print(f"    {k}: ES-AUC {100*v['es_auc'][0]:.2f} | meanPSD {100*v['mean_psd'][0]:.2f} | maxPSD {100*v['max_psd'][0]:.2f}")
