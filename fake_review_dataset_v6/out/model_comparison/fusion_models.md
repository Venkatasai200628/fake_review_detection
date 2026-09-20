# Phase A: fusion models compared (same 41 features, same splits)

Ranked by accuracy on test rows whose review text never appears in training (the column that is not inflated by template duplication).

| model | CV ROC (grouped) | test acc | **acc, unseen text** | F1 | ROC | PR-AUC@15% | edited photo | fake text | coord | false alarms |
|---|---|---|---|---|---|---|---|---|---|---|
| Extra Trees 1000, leaf 1 | 0.9553 | 0.9519 | **0.9519** | 0.9502 | 0.9780 | 0.9610 | 80.7% | 94.8% | 100.0% | 1.5% |
| Extra Trees 500 | 0.9551 | 0.9494 | **0.9494** | 0.9475 | 0.9792 | 0.9614 | 80.0% | 94.1% | 100.0% | 1.5% |
| Extra Trees 1000, leaf 2, sqrt | 0.9553 | 0.9494 | **0.9494** | 0.9475 | 0.9795 | 0.9628 | 80.0% | 94.1% | 100.0% | 1.5% |
| HistGradientBoosting | 0.9539 | 0.9444 | **0.9444** | 0.9437 | 0.9834 | 0.9628 | 83.0% | 96.3% | 100.0% | 4.2% |
| Soft vote ET+RF+HGB | 0.9550 | 0.9432 | **0.9432** | 0.9416 | 0.9801 | 0.9615 | 79.3% | 95.6% | 100.0% | 3.0% |
| Random Forest 500 (current) | 0.9530 | 0.9420 | **0.9420** | 0.9409 | 0.9780 | 0.9560 | 81.5% | 95.6% | 100.0% | 4.0% |
| XGBoost deep | 0.9548 | 0.9407 | **0.9407** | 0.9401 | 0.9831 | 0.9582 | 83.0% | 96.3% | 100.0% | 4.9% |
| XGBoost | 0.9535 | 0.9383 | **0.9383** | 0.9375 | 0.9834 | 0.9536 | 82.2% | 95.6% | 100.0% | 4.9% |
| Stacking (RF+XGB+HGB -> LR) | nan | 0.9383 | **0.9383** | 0.9380 | 0.9807 | 0.9622 | 83.0% | 97.0% | 100.0% | 5.7% |
| Logistic regression | 0.9433 | 0.9259 | **0.9259** | 0.9242 | 0.9740 | 0.9379 | 79.3% | 91.9% | 100.0% | 5.2% |