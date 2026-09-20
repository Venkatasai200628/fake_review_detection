# Phase A: fusion models compared (same 41 features, same splits)

Ranked by accuracy on test rows whose review text never appears in training (the column that is not inflated by template duplication).

| model | CV ROC (grouped) | test acc | **acc, unseen text** | F1 | ROC | PR-AUC@15% | edited photo | fake text | coord | false alarms |
|---|---|---|---|---|---|---|---|---|---|---|
| Extra Trees 1000, leaf 1 | 0.9409 | 0.9309 | **0.9309** | 0.9275 | 0.9644 | 0.9545 | 81.5% | 83.7% | 100.0% | 2.2% |
| Random Forest 500 (current) | 0.9386 | 0.9296 | **0.9296** | 0.9272 | 0.9612 | 0.9478 | 82.2% | 86.7% | 100.0% | 3.7% |
| Extra Trees 500 | 0.9419 | 0.9259 | **0.9259** | 0.9219 | 0.9653 | 0.9531 | 79.3% | 83.0% | 100.0% | 2.2% |
| Extra Trees 1000, leaf 2, sqrt | 0.9420 | 0.9259 | **0.9259** | 0.9219 | 0.9658 | 0.9541 | 79.3% | 83.0% | 100.0% | 2.2% |
| Soft vote ET+RF+HGB | 0.9412 | 0.9222 | **0.9222** | 0.9195 | 0.9641 | 0.9580 | 81.5% | 85.2% | 100.0% | 4.4% |
| HistGradientBoosting | 0.9325 | 0.9198 | **0.9198** | 0.9174 | 0.9652 | 0.9612 | 83.0% | 84.4% | 100.0% | 5.2% |
| Stacking (RF+XGB+HGB -> LR) | nan | 0.9198 | **0.9198** | 0.9178 | 0.9635 | 0.9524 | 82.2% | 86.7% | 100.0% | 5.7% |
| XGBoost | 0.9351 | 0.9185 | **0.9185** | 0.9165 | 0.9660 | 0.9516 | 82.2% | 85.9% | 100.0% | 5.7% |
| XGBoost deep | 0.9357 | 0.9173 | **0.9173** | 0.9157 | 0.9646 | 0.9502 | 83.0% | 86.7% | 100.0% | 6.4% |
| Logistic regression | 0.9302 | 0.9000 | **0.9000** | 0.8960 | 0.9631 | 0.9100 | 81.5% | 77.0% | 100.0% | 6.2% |