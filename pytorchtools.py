import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, **kwargs):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            skip_first_n_epochs (int): How many eopchs to skip at the beginning
                            Default: 0
        """
        # kwargs
        self.skip_first_n_epochs = kwargs.pop('skip_first_n_epochs', 0)
        self.times_called = 0
        
        if kwargs:
            print('Unknown kwarg', kwargs)
            raise
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def get_model_name(self, model):
        try:
            return model.model_name
        except AttributeError:
            return ''
    
    def __repr__(self):
        return f'EarlyStopping: patience={self.patience}, verbose:{self.verbose}'
        
    def __call__(self, val_loss, model):
        if self.times_called < self.skip_first_n_epochs:
            if self.verbose:
                print('\n\tSkipping the check for early stopping', end='')
            self.times_called += 1
            return
        
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'\n\tEarlyStopping counter: {self.counter} out of {self.patience}',
                 end='')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'\n\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...', end='')
        torch.save(model.state_dict(), self.get_model_name(model) + 'checkpoint.pt')
        self.val_loss_min = val_loss

