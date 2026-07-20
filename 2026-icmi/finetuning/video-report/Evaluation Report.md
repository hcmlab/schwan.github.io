# Qwen3-VL-8B Fine-Tuning Evaluation Report (ICEP-R)

This report provides a summary and analysis of the 5-fold cross-validation results for the fine-tuned Qwen3-VL-8B model on the ICEP-R (Infant and Caregiver Engagement Phases - Revised) dataset.

## 1. High-Level Summary

The fine-tuned Vision-Language Model achieved moderate success, with overall performance heavily skewed by extreme class imbalances in the dataset.

* **Infant Behaviors**: The model reached a **Micro F1-Score of 0.6441** across 9,921 samples.
* **Caregiver Behaviors**: The model performed worse on caregiver tracking, achieving a **Micro F1-Score of 0.5219**.
* **Core Issue**: The model successfully identifies obvious or highly frequent states (like "Background" or "None"), but largely fails to recognize nuanced or emotionally charged behaviors (like "Protest", "Positive", or "Intrusive").

---

## 2. Infant Results Analysis

The dataset is heavily dominated by `inon` (None/Passive) and `ineu` (Neutral).

| Category | Performance | Insights |
| :--- | :--- | :--- |
| **Excellent** (`bg`) | F1: 0.926 | Background scenes (no interaction) are visually distinct and easy for the VLM to classify. Precision is nearly perfect (0.988). |
| **Good** (`inon`) | F1: 0.757 | This is the majority class (N=5582). The model has a very high **Recall (0.834)**, implying it frequently defaults to guessing `inon` when it is unsure. |
| **Mediocre** (`ineu`) | F1: 0.452 | Neutral behavior is recognized with ~45% accuracy. It is likely being confused heavily with `inon`. |
| **Poor / Failing** (`ipos`, `ipro`, `iwit`, `iusc`) | F1: 0.000 - 0.268 | The model completely struggles with active emotional states. It only caught 18.9% of Protest (`ipro`) instances and a mere 7.2% of Positive (`ipos`) instances. It failed entirely on Withdrawn (`iwit`). |

---

## 3. Caregiver Results Analysis

Caregiver behaviors are significantly harder for the model to classify, reflecting a more even but challenging distribution among Neutral, Positive, and Physical Control states.

| Category | Performance | Insights |
| :--- | :--- | :--- |
| **Good** (`bg`) | F1: 0.817 | Similar to the infant track, empty/background frames are easily identified. |
| **Mediocre** (`cneu`, `cpos`, `cpvc`) | F1: 0.429 - 0.562 | The model is effectively guessing between these three dominant classes. Interestingly, Positive (`cpos`) has a high **Recall (0.642)** but low **Precision (0.405)**, meaning the model frequently hallucinates caregiver positivity when it isn't there. |
| **Failing** (`cint`, `cnon`, `cusc`) | F1: 0.000 - 0.111 | The model missed 100% of Intrusive (`cint`) and None (`cnon`) behaviors. Because these have very few examples (N=54 and N=46), the model never learned to recognize them. |

---

## 4. Key Takeaways & Recommendations

1. **Severe Class Imbalance Issue**: The most glaring issue is that the model defaults to majority classes. It learned to almost exclusively output `inon` (Infant None) or bounce between `cneu/cpvc` for caregivers, while minority classes score straight zeroes.
   * *Recommendation*: The training pipeline desperately needs a balancing mechanism. You should implement weighted loss functions, over-sample minority classes like `ipos`/`ipro`/`cint`, or under-sample majority classes like `inon`.

2. **Poor Affect/Emotion Recognition**: The Qwen-VL model is currently failing to read nuanced facial expressions or body language for states like "Protest" or "Positive", especially in infants.
   * *Recommendation*: The 8fps frame rate might be causing too much motion blur for subtle facial expressions. Try extracting frames at a higher quality, or fine-tuning the model specifically on static crops of the infant's face to strengthen its affect recognition.

3. **High False Positive Rate for Caregiver Warmth**: The model aggressively guesses `cpos` (Caregiver Positive), resulting in many false positives (Precision = 0.405). It seems to heavily associate any caregiver interaction with positivity.
