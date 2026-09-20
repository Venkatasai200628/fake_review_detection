# Is "fake text caught = 100%" real skill or template memorising?

Train 2490 reviews, test 810 reviews, text generated from templates.

## 1. Exact duplicate texts between test and train

- test texts that appear verbatim in train: **0 / 810 (0.0%)**
  - genuine: 0/405 (0.0%)
  - text_deception: 0/135 (0.0%)
  - coordinated_reuse: 0/135 (0.0%)
  - visual_manipulation: 0/135 (0.0%)

## 2. How close is each test text to its nearest training text? (TF-IDF cosine, 1.0 = identical)

| test group | mean nearest-neighbour similarity | median | share with sim > 0.9 |
|---|---|---|---|
| genuine | 0.585 | 0.564 | 0.7% |
| text_deception | 0.573 | 0.557 | 0.0% |
| coordinated_reuse | 0.599 | 0.590 | 0.0% |
| visual_manipulation | 0.571 | 0.561 | 0.0% |

## 3. The memorising floor: "copy the label of the most similar training review"

This rule has no learning in it at all. Whatever it scores is the part of the text task that is pure look-up.

- accuracy of the 1-nearest-neighbour copy rule: **0.768**
  - genuine: 78.5% correct
  - text_deception: 94.8% correct
  - coordinated_reuse: 100.0% correct
  - visual_manipulation: 30.4% correct

## 4. Vocabulary

- words used only in fake training reviews: 202
- words used only in genuine training reviews: 82
- shared: 602  (Jaccard 0.68)
