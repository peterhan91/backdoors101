import torch
from metrics.metric import Metric
from sklearn.metrics import cohen_kappa_score


class KappaMetric(Metric):

    def __init__(self):
        self.main_metric_name = 'Cohen Kappa'
        super().__init__(name='Kappa', train=False)

    def compute_metric(self, outputs: torch.Tensor, 
                        labels: torch.Tensor): 
        outputs[outputs < 0.5] = 0
        outputs[(outputs >= 0.5) & (outputs < 1.5)] = 1
        outputs[(outputs >= 1.5) & (outputs < 2.5)] = 2
        outputs[(outputs >= 2.5) & (outputs < 3.5)] = 3
        outputs[(outputs >= 3.5) & (outputs < 100)] = 4
        outputs = outputs.long().view(-1).detach().cpu().numpy()
        labels = labels.long().view(-1).detach().cpu().numpy()

        k = cohen_kappa_score(labels, outputs, weights='quadratic')
        # print(labels, outputs, k)
        return {'Cohen Kappa': k}