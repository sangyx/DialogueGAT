import random
import numpy as np
import torch


def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0, mode="max"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        if mode == "max":
            self.best_score = -np.Inf
            self.check_func = lambda x, y: x >= y
        else:
            self.best_score = np.inf
            self.check_func = lambda x, y: x <= y

    def __call__(self, score, model):
        if self.check_func(score, self.best_score + self.delta):
            # self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        """Saves model when score increase."""
        if self.verbose:
            print(
                f"Validation score increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...\n"
            )
        torch.save(model.state_dict(), self.save_path)
