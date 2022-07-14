import torch
from metrics.metric import Metric


class AccuracyMetric(Metric):

    def __init__(self, top_k=(1,)):
        self.top_k = top_k
        self.main_metric_name = 'Top-1'
        super().__init__(name='Accuracy', train=False)

    def compute_metric(self, outputs: torch.Tensor,
                       labels: torch.Tensor):
        """Computes the precision@k for the specified values of k"""
        # max_k = max(self.top_k)
        # batch_size = labels.shape[0]

        # _, pred = outputs.topk(max_k, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(labels.view(1, -1).expand_as(pred))

        # res = dict()
        # for k in self.top_k:
        #     correct_k = correct[:k].view(-1).float().sum(0)
        #     res[f'Top-{k}'] = (correct_k.mul_(100.0 / batch_size)).item()
        
        num_correct = 0
        num_samples = 0
        labels = labels.long().view(-1)
        outputs[outputs < 0.5] = 0
        outputs[(outputs >= 0.5) & (outputs < 1.5)] = 1
        outputs[(outputs >= 1.5) & (outputs < 2.5)] = 2
        outputs[(outputs >= 2.5) & (outputs < 3.5)] = 3
        outputs[(outputs >= 3.5) & (outputs < 100)] = 4
        outputs = outputs.long().view(-1)
        
        num_correct += (outputs == labels).sum()
        num_samples += outputs.shape[0]

        return {'Top-1': float(num_correct) / float(num_samples) * 100}
