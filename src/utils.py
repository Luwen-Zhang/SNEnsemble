import numpy as np
import torch

name_mapping = {
    'wt.C': 'wt.C',
    'wt.Si': 'wt.Si',
    'wt.Mn': 'wt.Mn',
    'wt.P': 'wt.P',
    'wt.S': 'wt.S',
    'wt.Ni': 'wt.Ni',
    'wt.Cr': 'wt.Cr',
    'wt.Mo': 'wt.Mo',
    'wt.N': 'wt.N',
    'temperature(celsius)': 'temperature',
    'strain_amplitude(%)': 'strain amplitude',
    'hold_time(h)': 'hold time',
    'strain_rate(s-1)': 'strain rate',
    'fatigue_life': 'fatigue life'
}

def replace_column_name(df):
    columns = list(df.columns)
    for idx in range(len(columns)):
        try:
            columns[idx] = name_mapping[columns[idx]]
        except:
            pass
    df.columns = columns
    return df

def plot_truth_pred(ax,ground_truth,prediction,**kargs):
    ax.scatter(ground_truth, prediction,**kargs)
    ax.set_xlabel('Ground truth')
    ax.set_ylabel('Prediction')


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss