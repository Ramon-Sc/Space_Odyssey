"""
Linear Regression on collected features
X: features 1xN
y: margin values 1x1
margins refers to fucntional margins of the DNN classifier:
max(logits(x)) - sorted(logits(x))[1] #second highest logit 
"""


class regression:
    def __init__(self, config):
        self.config = config

    def linear_regression(self, X, y):
        pass
