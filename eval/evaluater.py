import os
import contextlib
import copy
import numpy as np


from data import ReactionImageData


def convert_to_xywh(box, width, height):
    xmin, ymin, xmax, ymax = box
    return [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height]


EMPTY_STATS = {'gold_hits': 0, 'gold_total': 0, 'pred_hits': 0, 'pred_total': 0, 'image': 0}


class ReactionEvaluator(object):

    def evaluate_image(self, gold_image, pred_image, **kwargs):
        data = ReactionImageData(gold_image, pred_image)
        return data.evaluate(**kwargs)

    def compute_metrics(self, gold_hits, gold_total, pred_hits, pred_total):
        # 特殊情况：标准答案为空
        if gold_total == 0:
            if pred_total == 0:
                # 标准答案为空，预测也为空，完全正确
                return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            else:
                # 标准答案为空，但预测了东西，完全错误
                return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}
        
        # 特殊情况：预测为空但标准答案不为空
        if pred_total == 0:
            return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0}
        
        # 正常情况
        precision = pred_hits / pred_total
        recall = gold_hits / gold_total
        f1 = precision * recall * 2 / max(precision + recall, 1e-6)
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def evaluate(self, groundtruths, predictions, **kwargs):
        gold_hits, gold_total, pred_hits, pred_total = 0, 0, 0, 0
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image, **kwargs)
            gold_hits += sum(gh)
            gold_total += len(gh)
            pred_hits += sum(ph)
            pred_total += len(ph)
        return self.compute_metrics(gold_hits, gold_total, pred_hits, pred_total)

    def evaluate_by_size(self, groundtruths, predictions, **kwargs):
        group_stats = {}
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image, **kwargs)
            gtotal = len(gh)
            if gtotal not in group_stats:
                group_stats[gtotal] = copy.deepcopy(EMPTY_STATS)
            group_stats[gtotal]['gold_hits'] += sum(gh)
            group_stats[gtotal]['gold_total'] += len(gh)
            group_stats[gtotal]['pred_hits'] += sum(ph)
            group_stats[gtotal]['pred_total'] += len(ph)
            group_stats[gtotal]['image'] += 1
        group_scores = {}
        for gtotal, stats in group_stats.items():
            group_scores[gtotal] = self.compute_metrics(
                stats['gold_hits'], stats['gold_total'], stats['pred_hits'], stats['pred_total'])
        return group_scores, group_stats

    def evaluate_by_group(self, groundtruths, predictions, **kwargs):
        group_stats = {}
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image, **kwargs)
            diagram_type = gold_image['diagram_type']
            if diagram_type not in group_stats:
                group_stats[diagram_type] = copy.deepcopy(EMPTY_STATS)
            group_stats[diagram_type]['gold_hits'] += sum(gh)
            group_stats[diagram_type]['gold_total'] += len(gh)
            group_stats[diagram_type]['pred_hits'] += sum(ph)
            group_stats[diagram_type]['pred_total'] += len(ph)
            group_stats[diagram_type]['image'] += 1
        group_scores = {}
        for group, stats in group_stats.items():
            group_scores[group] = self.compute_metrics(
                stats['gold_hits'], stats['gold_total'], stats['pred_hits'], stats['pred_total'])
        return group_scores, group_stats

    def evaluate_summarize(self, groundtruths, predictions, **kwargs):
        size_scores, size_stats = self.evaluate_by_size(groundtruths, predictions, **kwargs)
        summarize = {
            'overall': copy.deepcopy(EMPTY_STATS),
            # 'single': copy.deepcopy(EMPTY_STATS),
            # 'multiple': copy.deepcopy(EMPTY_STATS)
        }
        for size, stats in size_stats.items():
            if type(size) is int:
                # output = summarize['single'] if size <= 1 else summarize['multiple']
                for key in stats:
                    # output[key] += stats[key]
                    summarize['overall'][key] += stats[key]
        scores = {}
        for key, val in summarize.items():
            scores[key] = self.compute_metrics(val['gold_hits'], val['gold_total'], val['pred_hits'], val['pred_total'])
        return scores, summarize, size_stats

    def evaluate_per_image(self, groundtruths, predictions, **kwargs):
        """
        计算每张图片的召回率、精确率和F1分数
        
        Returns:
            list: 每张图片的评估指标字典列表，包含 'recall', 'precision', 'f1', 'gold_total', 'pred_total'
        """
        per_image_metrics = []
        for gold_image, pred_image in zip(groundtruths, predictions):
            gh, ph = self.evaluate_image(gold_image, pred_image, **kwargs)
            metrics = self.compute_metrics(sum(gh), len(gh), sum(ph), len(ph))
            metrics['gold_total'] = len(gh)
            metrics['pred_total'] = len(ph)
            metrics['file_name'] = gold_image.get('file_name', 'unknown')
            per_image_metrics.append(metrics)
        return per_image_metrics
