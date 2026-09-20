# Phase A: fusion models compared (same 41 features, same splits)

Ranked by accuracy on test rows whose review text never appears in training (the column that is not inflated by template duplication).

| model | CV ROC (grouped) | test acc | **acc, unseen text** | F1 | ROC | PR-AUC@15% | edited photo | fake text | coord | false alarms |
|---|---|---|---|---|---|---|---|---|---|---|
| Extra Trees 1000, leaf 2, sqrt | 0.9419 | 0.9321 | **0.9321** | 0.9287 | 0.9668 | 0.9547 | 80.7% | 84.4% | 100.0% | 2.0% |
| Random Forest 500 (current) | 0.9375 | 0.9296 | **0.9296** | 0.9274 | 0.9629 | 0.9530 | 83.0% | 86.7% | 100.0% | 4.0% |
| Extra Trees 500 | 0.9420 | 0.9296 | **0.9296** | 0.9261 | 0.9662 | 0.9540 | 80.0% | 84.4% | 100.0% | 2.2% |
| Extra Trees 1000, leaf 1 | 0.9404 | 0.9296 | **0.9296** | 0.9259 | 0.9668 | 0.9558 | 79.3% | 84.4% | 100.0% | 2.0% |
| Soft vote ET+RF+HGB | 0.9415 | 0.9272 | **0.9272** | 0.9245 | 0.9659 | 0.9605 | 82.2% | 85.2% | 100.0% | 3.7% |
| HistGradientBoosting | 0.9350 | 0.9247 | **0.9247** | 0.9227 | 0.9675 | 0.9664 | 83.7% | 85.9% | 100.0% | 4.9% |
| Stacking (RF+XGB+HGB -> LR) | nan | 0.9222 | **0.9222** | 0.9202 | 0.9657 | 0.9585 | 83.0% | 85.9% | 100.0% | 5.2% |
| XGBoost deep | 0.9359 | 0.9210 | **0.9210** | 0.9190 | 0.9689 | 0.9628 | 83.0% | 85.9% | 100.0% | 5.4% |
| XGBoost | 0.9359 | 0.9173 | **0.9173** | 0.9151 | 0.9707 | 0.9647 | 81.5% | 85.9% | 100.0% | 5.7% |
| Logistic regression | 0.9312 | 0.9000 | **0.9000** | 0.8963 | 0.9627 | 0.9069 | 80.0% | 79.3% | 100.0% | 6.4% |