# Comparative Evaluation Report: ICEP Audio Joint Annotation

## Experimental Setup

**Model:** `qwen2_audio_7b`  
**Profile:** `slurm_a40`  
**Dataset:** `icep_no_bg_audio_joint`  
**Role mode:** `joint`

This report summarizes three evaluation scenarios for scientific reporting:

1. **Test 1: 5-fold CV and fold-0 test (1737-sample fold-0)**  
   This setting reports the fold-0 evaluation result from a 5-fold setup without a separate model-selection stage.

2. **Test 2: Model-selection evaluation on the train\_dev portion (1580-sample fold-0)**  
   Training on train\_dev folds, validating on each held-out CV fold, and aggregating those fold results.  
   This reflects how the training recipe performed during model selection on the train\_dev partition.

3. **Test 3: Final leave-out test set evaluation**  
   Training one final model on the full train\_dev split, then predicting once on the untouched test split.  
   This reflects final generalization to unseen test data.

---

## 1. Overall Performance Comparison

| Scenario | Samples | Infant Micro F1 | Infant Weighted F1 | Caregiver Micro F1 | Caregiver Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Test 1: 5-fold CV and fold-0 test | 1737 | 0.6707 | 0.5780 | 0.6799 | 0.6721 |
| Test 2: Model-selection evaluation on train_dev | 1580 | 0.6791 | 0.5728 | 0.6892 | 0.6807 |
| Test 3: Final leave-out test set | 1818 | 0.6337 | 0.5473 | 0.6529 | 0.6421 |

---

## 2. Test 1: 5-fold CV and Fold-0 Test (1737-sample fold-0)

### Fold Summary

| Fold | Val Samples | Parse Errors | Invalid Predictions | Infant Weighted F1 | Caregiver Weighted F1 |
|---|---:|---:|---:|---:|---:|
| 0 | 1737 | 0 | 0 | 0.5780 | 0.6721 |

### Infant Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ineu | 0.5000 | 0.0087 | 0.0170 | 346 |
| inon | 0.6694 | 0.9540 | 0.7867 | 1044 |
| ipos | 0.2353 | 0.0482 | 0.0800 | 83 |
| ipro | 0.7168 | 0.6480 | 0.6807 | 250 |
| iusc | 0.0000 | 0.0000 | 0.0000 | 14 |
| **Micro Avg** | **0.6707** | **0.6707** | **0.6707** | **1737** |
| **Weighted Avg** | **0.6163** | **0.6707** | **0.5780** | **1737** |

### Caregiver Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cint | 0.0000 | 0.0000 | 0.0000 | 2 |
| cneu | 0.6149 | 0.8783 | 0.7234 | 600 |
| cnon | 0.0000 | 0.0000 | 0.0000 | 2 |
| cpos | 0.4783 | 0.3529 | 0.4062 | 374 |
| cpvc | 0.8642 | 0.6877 | 0.7660 | 759 |
| **Micro Avg** | **0.6799** | **0.6799** | **0.6799** | **1737** |
| **Weighted Avg** | **0.6930** | **0.6799** | **0.6721** | **1737** |

### Infant Confusion Matrix

| True \ Pred | ineu | inon | ipos | ipro | iusc |
|---|---:|---:|---:|---:|---:|
| ineu | 3 | 320 | 5 | 18 | 0 |
| inon | 1 | 996 | 4 | 43 | 0 |
| ipos | 2 | 74 | 4 | 3 | 0 |
| ipro | 0 | 84 | 4 | 162 | 0 |
| iusc | 0 | 14 | 0 | 0 | 0 |

### Caregiver Confusion Matrix

| True \ Pred | cint | cneu | cnon | cpos | cpvc |
|---|---:|---:|---:|---:|---:|
| cint | 0 | 0 | 0 | 1 | 1 |
| cneu | 0 | 527 | 0 | 49 | 24 |
| cnon | 0 | 1 | 0 | 0 | 1 |
| cpos | 0 | 186 | 0 | 132 | 56 |
| cpvc | 0 | 143 | 0 | 94 | 522 |

---

## 3. Test 2: Model-Selection Evaluation on Train_Dev (1580-sample fold-0)

**Protocol:** training on train_dev folds, validating on each held-out CV fold, and aggregating fold-level results.  
This result characterizes recipe performance during model selection on the train_dev portion.

### Summary

| Samples | Parse Errors | Invalid Predictions | Infant Weighted F1 | Caregiver Weighted F1 |
|---:|---:|---:|---:|---:|
| 1580 | 0 | 0 | 0.5728 | 0.6807 |

### Infant Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ineu | 1.0000 | 0.0102 | 0.0202 | 392 |
| inon | 0.6529 | 0.9869 | 0.7859 | 913 |
| ipos | 0.9048 | 0.2043 | 0.3333 | 93 |
| ipro | 0.8514 | 0.8563 | 0.8539 | 174 |
| iusc | 0.0000 | 0.0000 | 0.0000 | 8 |
| **Micro Avg** | **0.6791** | **0.6791** | **0.6791** | **1580** |
| **Weighted Avg** | **0.7724** | **0.6791** | **0.5728** | **1580** |

### Caregiver Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cint | 0.0000 | 0.0000 | 0.0000 | 14 |
| cneu | 0.5829 | 0.9184 | 0.7132 | 490 |
| cnon | 0.0000 | 0.0000 | 0.0000 | 5 |
| cpos | 0.6335 | 0.4454 | 0.5230 | 357 |
| cpvc | 0.8618 | 0.6723 | 0.7553 | 714 |
| **Micro Avg** | **0.6892** | **0.6892** | **0.6892** | **1580** |
| **Weighted Avg** | **0.7134** | **0.6893** | **0.6807** | **1580** |

### Infant Confusion Matrix

| True \ Pred | ineu | inon | ipos | ipro |
|---|---:|---:|---:|---:|
| ineu | 4 | 380 | 0 | 8 |
| inon | 0 | 901 | 2 | 10 |
| ipos | 0 | 67 | 19 | 7 |
| ipro | 0 | 25 | 0 | 149 |
| iusc | 0 | 7 | 0 | 1 |

### Caregiver Confusion Matrix

| True \ Pred | cneu | cpos | cpvc |
|---|---:|---:|---:|
| cint | 7 | 1 | 6 |
| cneu | 450 | 20 | 20 |
| cnon | 2 | 2 | 1 |
| cpos | 148 | 159 | 50 |
| cpvc | 165 | 69 | 480 |

---

## 4. Test 3: Final Leave-Out Test Set Evaluation

**Protocol:** training one final model on the full train_dev split, then evaluating once on the untouched test split.  
This score reflects final performance on unseen test data.

### Summary

| Samples | Parse Errors | Invalid Predictions | Infant Weighted F1 | Caregiver Weighted F1 |
|---:|---:|---:|---:|---:|
| 1818 | 0 | 0 | 0.5473 | 0.6421 |

### Infant Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ineu | 0.2838 | 0.0547 | 0.0917 | 384 |
| inon | 0.6532 | 0.9279 | 0.7667 | 1082 |
| ipos | 0.2857 | 0.0282 | 0.0513 | 142 |
| ipro | 0.6373 | 0.6373 | 0.6373 | 193 |
| iusc | 0.0000 | 0.0000 | 0.0000 | 14 |
| iwit | 0.0000 | 0.0000 | 0.0000 | 3 |
| **Micro Avg** | **0.6337** | **0.6337** | **0.6337** | **1818** |
| **Weighted Avg** | **0.5387** | **0.6337** | **0.5473** | **1818** |

### Caregiver Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cint | 0.0000 | 0.0000 | 0.0000 | 6 |
| cneu | 0.6132 | 0.8173 | 0.7007 | 646 |
| cnon | 0.0000 | 0.0000 | 0.0000 | 16 |
| cpos | 0.4491 | 0.3432 | 0.3891 | 373 |
| cpvc | 0.7902 | 0.6834 | 0.7329 | 777 |
| **Micro Avg** | **0.6529** | **0.6529** | **0.6529** | **1818** |
| **Weighted Avg** | **0.6478** | **0.6529** | **0.6421** | **1818** |

### Infant Confusion Matrix

| True \ Pred | ineu | inon | ipos | ipro |
|---|---:|---:|---:|---:|
| ineu | 21 | 343 | 1 | 19 |
| inon | 33 | 1004 | 6 | 39 |
| ipos | 11 | 115 | 4 | 12 |
| ipro | 9 | 58 | 3 | 123 |
| iusc | 0 | 14 | 0 | 0 |
| iwit | 0 | 3 | 0 | 0 |

### Caregiver Confusion Matrix

| True \ Pred | cneu | cpos | cpvc |
|---|---:|---:|---:|
| cint | 3 | 1 | 2 |
| cneu | 528 | 65 | 53 |
| cnon | 13 | 1 | 2 |
| cpos | 161 | 128 | 84 |
| cpvc | 156 | 90 | 531 |

---

## 5. Comparative Interpretation

Several consistent patterns emerge across all three scenarios:

- The strongest infant class is **`inon`**, which achieves the highest recall across all evaluations.
- The strongest caregiver classes are **`cneu`** and **`cpvc`**.
- Rare classes such as **`iusc`**, **`iwit`**, **`cint`**, and **`cnon`** remain unrecognized or nearly unrecognized, suggesting strong class imbalance effects.
- The model-selection setting on train_dev yields slightly higher scores than the final untouched leave-out test set, which is expected because the final test split is fully unseen.
- The 1737-sample fold-0 scenario shows competitive caregiver performance and slightly better infant weighted F1 than the model-selection scenario, but all three settings exhibit the same general confusion pattern: rare behaviors are often collapsed into dominant categories.

### Main confusion trends

- **Infant:** minority classes are frequently predicted as `inon`.
- **Caregiver:** `cpos`, `cint`, and `cnon` are often confused with `cneu` or `cpvc`.

---

## 6. Publication-Style Summary Paragraph

For the ICEP no-background audio joint annotation setting, `qwen2_audio_7b` was evaluated under three scenarios: a fold-0 evaluation from a 5-fold setup (1737 samples), a model-selection cross-validation evaluation on the train_dev portion (1580 samples), and a final evaluation on an unseen leave-out test set (1818 samples). The fold-0 setting achieved weighted F1 scores of **0.5780** for infant and **0.6721** for caregiver annotations. The train_dev model-selection evaluation yielded weighted F1 scores of **0.5728** and **0.6807**, respectively. Final evaluation on the untouched leave-out test set resulted in weighted F1 scores of **0.5473** for infant and **0.6421** for caregiver annotations. Across all settings, the model performed best on dominant classes such as `inon`, `cneu`, and `cpvc`, while minority classes including `iusc`, `iwit`, `cint`, and `cnon` remained poorly recognized, indicating persistent class imbalance and a tendency to map rare behaviors into more frequent categories.
