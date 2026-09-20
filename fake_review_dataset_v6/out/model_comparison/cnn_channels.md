# Phase B: which forensic maps should the edit detector see?

Same CNN, same training rows, only the input channels change. Each row is the average of 2 seeds; the two individual seed scores are shown so you can see whether a difference is bigger than seed noise.

| input channels | ROC (test photos) | seed 1 / seed 2 | copy-move | splice | noise | caught @5% false alarms |
|---|---|---|---|---|---|---|
| 3 ch: ELA, ELA-std, noise (current) | **0.9442** | 0.950 / 0.906 | 0.886 | 0.971 | 0.961 | 70.4% |
| 6 ch: all (+ copy-move votes) | **0.9367** | 0.919 / 0.940 | 0.858 | 0.957 | 0.976 | 70.4% |