# Quality report: medium_20261026_234912

## coding

280 runs; runs with rounds 1..t complete: {'1': 280, '2': 280, '3': 280, '4': 280, '5': 280}

7280 replies saved, 7515033 generated tokens, 1343 gen tok/s overall (first to last reply)

### Call health

| phase | agent | calls | failed_% | retried_% | truncated_% | prompt_too_long | think_tok_mean | gen_tok_mean | gen_tok_p95 |
|---|---|---|---|---|---|---|---|---|---|
| init | generator | 280 | 0.0 | 0.0 | 0.0 | 0 | 461.7 | 633.3 | 1129.5 |
| orch | generator | 1400 | 0.0 | 0.1 | 0.0 | 0 | 758.7 | 1344.7 | 3473.1 |
| review | correctness | 1400 | 0.0 | 0.0 | 0.0 | 0 | 1177.4 | 1374.9 | 2779.1 |
| review | optimality | 1400 | 0.0 | 0.1 | 0.0 | 0 | 584.4 | 764.1 | 1324.0 |
| review | readability | 1400 | 0.0 | 0.0 | 0.0 | 0 | 642.3 | 829.6 | 1379.3 |
| review | security | 1400 | 0.0 | 0.0 | 0.0 | 0 | 725.9 | 928.0 | 1683.5 |


### Score sanity

| agent | mean by round | std by round | score counts |
|---|---|---|---|
| security | [2.0, 4.09, 4.58, 4.78, 4.85] | [1.18, 0.73, 0.55, 0.44, 0.37] | {1: 130, 2: 85, 3: 76, 4: 390, 5: 719} |
| optimality | [4.01, 4.75, 4.9, 4.98, 4.98] | [0.85, 0.44, 0.33, 0.14, 0.13] | {1: 2, 2: 18, 3: 38, 4: 246, 5: 1096} |
| correctness | [3.22, 4.46, 4.75, 4.93, 4.94] | [1.06, 0.61, 0.63, 0.36, 0.34] | {1: 18, 2: 77, 3: 76, 4: 298, 5: 931} |
| readability | [3.66, 4.53, 4.88, 4.95, 4.99] | [0.58, 0.51, 0.36, 0.23, 0.1] | {1: 1, 2: 10, 3: 81, 4: 360, 5: 948} |

Mean max-min spread across agents by round: [2.36, 0.89, 0.54, 0.29, 0.2]  
Share of scores at the scale endpoints by round: [0.236, 0.524, 0.808, 0.918, 0.947]


### Generated code validity

| round | codes | empty | python_parses_% | cpp_syntax_ok_% (sample) | mean_chars |
|---|---|---|---|---|---|
| 1 | 280 | 0 | 99.0 | 96.2 | 547.5 |
| 2 | 280 | 0 | 98.5 | 96.2 | 1685.2 |
| 3 | 280 | 0 | 99.0 | 95.0 | 2089.4 |
| 4 | 280 | 0 | 99.0 | 93.8 | 2225.9 |
| 5 | 280 | 0 | 99.5 | 92.5 | 2291.8 |
| 6 | 280 | 0 | 99.0 | 93.8 | 2316.9 |


CodeQL: 1680 codes analysed, 37.4% have at least one alert.

## hiring

640 runs; runs with rounds 1..t complete: {'1': 640, '2': 640, '3': 640, '4': 640, '5': 640}

12800 replies saved, 18710042 generated tokens, 1327 gen tok/s overall (first to last reply)

### Call health

| phase | agent | calls | failed_% | retried_% | truncated_% | prompt_too_long | think_tok_mean | gen_tok_mean | gen_tok_p95 |
|---|---|---|---|---|---|---|---|---|---|
| review | experience | 3200 | 0.0 | 0.0 | 0.0 | 0 | 1144.3 | 1471.2 | 2444.3 |
| review | impact | 3200 | 0.0 | 0.0 | 0.0 | 0 | 1106.9 | 1417.1 | 2353.0 |
| review | operational | 3200 | 0.0 | 0.0 | 0.0 | 0 | 1135.7 | 1470.8 | 2452.0 |
| review | subject | 3200 | 0.0 | 0.0 | 0.0 | 0 | 1155.5 | 1487.8 | 2563.1 |


### Score sanity

| agent | mean by round | std by round | score counts |
|---|---|---|---|
| subject | [3.3, 3.32, 3.33, 3.33, 3.33] | [0.75, 0.76, 0.77, 0.77, 0.77] | {1: 5, 2: 543, 3: 1085, 4: 1546, 5: 21} |
| experience | [3.35, 3.36, 3.36, 3.36, 3.36] | [0.84, 0.83, 0.83, 0.82, 0.82] | {1: 1, 2: 587, 3: 1017, 4: 1455, 5: 140} |
| impact | [3.13, 3.22, 3.26, 3.28, 3.28] | [0.76, 0.77, 0.77, 0.78, 0.78] | {2: 644, 3: 1181, 4: 1349, 5: 26} |
| operational | [3.2, 3.26, 3.29, 3.29, 3.29] | [0.79, 0.8, 0.8, 0.8, 0.8] | {2: 686, 3: 994, 4: 1497, 5: 23} |

Mean max-min spread across agents by round: [0.53, 0.31, 0.24, 0.22, 0.22]  
Share of scores at the scale endpoints by round: [0.014, 0.017, 0.018, 0.018, 0.018]


### Adversary check (mean score by round)

- adv | evaluator that is adversarial in 'adv': [3.225, 3.294, 3.344, 3.353, 3.35]
- adv | other evaluators: [3.264, 3.305, 3.322, 3.325, 3.326]
- base | evaluator that is adversarial in 'adv': [3.197, 3.256, 3.284, 3.288, 3.288]
- base | other evaluators: [3.25, 3.291, 3.299, 3.304, 3.308]

## review

300 runs; runs with rounds 1..t complete: {'1': 300, '2': 300, '3': 191, '4': 0, '5': 0}

3577 replies saved, 14982514 generated tokens, 561 gen tok/s overall (first to last reply)

### Call health

| phase | agent | calls | failed_% | retried_% | truncated_% | prompt_too_long | think_tok_mean | gen_tok_mean | gen_tok_p95 |
|---|---|---|---|---|---|---|---|---|---|
| orch | author | 600 | 0.0 | 1.3 | 0.0 | 0 | 1686.8 | 6004.3 | 8485.5 |
| orch | metareviewer | 600 | 0.0 | 0.0 | 0.0 | 0 | 1109.2 | 2889.9 | 3546.7 |
| review | empiricist | 792 | 0.0 | 0.0 | 0.0 | 0 | 1728.3 | 4008.1 | 4918.8 |
| review | pragmatist | 794 | 0.0 | 0.1 | 0.0 | 0 | 1629.8 | 3882.9 | 4779.8 |
| review | theorist | 791 | 0.0 | 0.0 | 0.0 | 0 | 1910.2 | 4283.9 | 5648.5 |


### Score sanity

| agent | mean by round | std by round | score counts |
|---|---|---|---|
| pragmatist | [5.51, 6.05, 6.14] | [0.97, 0.96, 0.96] | {1: 3, 2: 3, 3: 14, 4: 14, 5: 2, 6: 1, 7: 1, 8: 11} |
| theorist | [5.44, 5.98, 6.13] | [1.02, 0.93, 0.92] | {2: 6, 3: 10, 4: 29, 5: 127, 6: 1, 7: 140, 8: 6} |
| empiricist | [5.3, 5.9, 6.11] | [0.91, 0.96, 0.94] | {1: 3, 2: 3, 3: 12, 4: 25, 5: 1, 6: 3, 7: 109, 8: 6} |

Mean max-min spread across agents by round: [0.57, 0.43, 0.28]  
Share of scores at the scale endpoints by round: [0.003, 0.003, 0.003]


## Homogenization metrics (from summary.json)

```json
{
 "coverage": {
  "coding": {
   "runs_total": 280,
   "runs_analysed": 280,
   "rounds_analysed": 5
  },
  "hiring_base": {
   "runs_total": 320,
   "runs_analysed": 320,
   "rounds_analysed": 5
  },
  "hiring_adv": {
   "runs_total": 320,
   "runs_analysed": 320,
   "rounds_analysed": 5
  },
  "review": {
   "runs_total": 300,
   "runs_analysed": 194,
   "rounds_analysed": 3
  }
 },
 "coding": {
  "runs": 280,
  "definition_rates": {
   "conformity": 0.45714285714285713,
   "polarization": 0.5375,
   "inertia": 0.5857142857142857
  },
  "conformity": [
   1.040625,
   0.22723214285714285,
   0.13995535714285715,
   0.06495535714285715,
   0.05133928571428571
  ],
  "polarization": [
   1.4428571428571428,
   0.5392857142857143,
   0.21875,
   0.08839285714285715,
   0.05892857142857143
  ],
  "inertia": [
   1.2446428571428572,
   0.35982142857142857,
   0.15892857142857142,
   0.06160714285714286
  ]
 },
 "hiring_base": {
  "runs": 320,
  "definition_rates": {
   "conformity": 0.0875,
   "polarization": 0.0625,
   "inertia": 0.109375
  },
  "conformity": [
   0.1060546875,
   0.0576171875,
   0.048828125,
   0.04609375,
   0.044921875
  ],
  "polarization": [
   1.34921875,
   1.32421875,
   1.3171875,
   1.3171875,
   1.3140625
  ],
  "inertia": [
   0.075,
   0.01484375,
   0.0046875,
   0.003125
  ]
 },
 "hiring_adv": {
  "runs": 320,
  "definition_rates": {
   "conformity": 0.1015625,
   "polarization": 0.0859375,
   "inertia": 0.13125
  },
  "conformity": [
   0.1123046875,
   0.0697265625,
   0.0494140625,
   0.0455078125,
   0.0435546875
  ],
  "polarization": [
   1.35859375,
   1.31171875,
   1.30234375,
   1.29921875,
   1.29609375
  ],
  "inertia": [
   0.075,
   0.0296875,
   0.0078125,
   0.003125
  ]
 },
 "review": {
  "runs": 194,
  "definition_rates": {
   "conformity": 0.29381443298969073,
   "polarization": 0.42783505154639173,
   "inertia": 0.7628865979381443
  },
  "conformity": [
   0.12886597938144304,
   0.09450171821305825,
   0.06328751431844208
  ],
  "polarization": [
   3.819587628865978,
   3.6752577319587623,
   3.5919243986254297
  ],
  "inertia": [
   0.5618556701030927,
   0.1417525773195875
  ]
 }
}
```
