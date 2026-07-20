"""ClassificationStats extracted from scripts/stat_analysis.py for standalone use on SLURM."""

from collections import defaultdict


class ClassificationStats:
    def __init__(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.support = defaultdict(int)
        self.total_predictions = defaultdict(int)
        self.all_classes = set()
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

    def update(self, true_label, pred_label):
        if not true_label: true_label = "(MISSING)"
        if not pred_label: pred_label = "(MISSING)"

        true_label = true_label.lower().strip()
        pred_label = pred_label.lower().strip()

        if true_label != "(missing)": self.all_classes.add(true_label)
        if pred_label != "(missing)": self.all_classes.add(pred_label)

        if true_label != "(missing)":
            self.support[true_label] += 1
        if pred_label != "(missing)":
            self.total_predictions[pred_label] += 1

        self.confusion_matrix[true_label][pred_label] += 1

        if true_label == pred_label:
            if true_label != "(missing)":
                self.tp[true_label] += 1
        else:
            if true_label != "(missing)":
                self.fn[true_label] += 1
            if pred_label != "(missing)":
                self.fp[pred_label] += 1

    def get_metrics(self):
        metrics = {}
        sorted_classes = sorted(list(self.all_classes))

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for label in sorted_classes:
            if label == "(missing)": continue
            tp = self.tp[label]
            fp = self.fp[label]
            fn = self.fn[label]

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": self.support[label]
            }

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        metrics["_MICRO_AVG"] = {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1_score": round(micro_f1, 4),
            "support": sum(self.support.values())
        }

        total_support = sum(self.support.values())
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

        if total_support > 0:
            for label, m in metrics.items():
                if label.startswith("_"): continue
                s = m["support"]
                weighted_precision += m["precision"] * s
                weighted_recall += m["recall"] * s
                weighted_f1 += m["f1_score"] * s

            metrics["_WEIGHTED_AVG"] = {
                "precision": round(weighted_precision / total_support, 4),
                "recall": round(weighted_recall / total_support, 4),
                "f1_score": round(weighted_f1 / total_support, 4),
                "support": total_support
            }

        return metrics

    def get_confusion_matrix_dict(self):
        cm = {}
        labels = sorted(list(self.all_classes) + ["(missing)"])
        for true_lbl in labels:
            row = {}
            for pred_lbl in labels:
                val = self.confusion_matrix[true_lbl][pred_lbl]
                if val > 0:
                    row[pred_lbl] = val
            if row:
                cm[true_lbl] = row
        return cm

    def to_dict(self):
        return {
            "metrics": self.get_metrics(),
            "confusion_matrix": self.get_confusion_matrix_dict()
        }

    def print_report(self, title="Classification Report"):
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}")
        print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print(f"{'-'*80}")

        metrics = self.get_metrics()

        for label in sorted(list(self.all_classes)):
            if label not in metrics: continue
            m = metrics[label]
            print(f"{label:<20} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']:<8}")

        print(f"{'-'*80}")
        if "_MICRO_AVG" in metrics:
            m = metrics["_MICRO_AVG"]
            print(f"{'MICRO AVG':<20} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']:<8}")

        if "_WEIGHTED_AVG" in metrics:
            m = metrics["_WEIGHTED_AVG"]
            print(f"{'WEIGHTED AVG':<20} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']:<8}")

        print(f"{'='*80}\n")
