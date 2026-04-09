import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader

#modify loss function
from torch.nn import KLDivLoss
import torch.nn as nn

class TabNetClassifier(TabModel):
    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'

    def weight_updater(self, weights):
        """
        Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
         #print(f"y_pred and y_true must be lists of same length, got {len(y_pred)} and {len(y_true)}")
        # print(y_pred.shape)
        # print(y_true.shape)
        #negative log-likelihood loss function
        m = nn.LogSoftmax(dim=1)
        nll_loss=nn.NLLLoss()
        lossone=nll_loss(m(y_pred),y_true.long())
        return lossone
        # # losstwo=self.loss_fn(y_pred, y_true.long())
        # # loss=torch.mean(lossone + losstwo)
       
        # return lossone
        #y_prednwew=softmax()
        #KL div loss function
        # y = torch.eye(16).cuda() 
        # y_truen=y[y_true.long()]
        # kl_loss = KLDivLoss(reduction = 'batchmean')
        # outputone = kl_loss(y_pred,  y_truen)
      
        # return outputone
        # y_prednwew = torch.nn.Softmax(dim=1)(y_pred).cpu().detach().numpy()
    
        # #return self.loss_fn(y_pred, y_true.long())
        # kl_loss = KLDivLoss(reduction = 'batchmean')
        # lossone = kl_loss(y_prednwew,  y_true.long())
        # return outputone
        # #oldloss=torch.nn.functional.cross_entropy
        ##outputtwo=oldloss(y_pred, y_true.long())
        # #print(f"starting executing of new loss function")
        # self.loss_fn=outputone
        
        # output=y_pred
        # target=y_true.long()
        # squares = output ** 2 # (x ^ 2, y ^ 2)
        # loss_1 = ((squares[:, ::2] + squares[:, 1::2]) - 1) ** 2 # (x ^ 2 + y ^ 2 - 1) ** 2

    # # Compute the second loss, 1 - cos
        # loss_2 =  1. - torch.cos(torch.atan2(output[:, 1::2], output[:, ::2]) - target)  
        # self.loss_fn=torch.mean(loss_1 + loss_2)
        #return
        #return cosine(y_pred, y_true.long())
        
        #return self.loss_fn(y_pred, y_true.long())
        #m = nn.LogSoftmax(dim=1)
        #nll_loss = nn.NLLLoss()
        #output = nll_loss(m(y_pred), y_true.long())
        #outputloss = 1*outputone+1*outputtwo
        

       # return outputloss

    # def compute_loss(self, y_pred, y_true):

        # '''

        # Cosine loss function. Penalizes the area out of the unit circle and wraps the output around 2pi.

        # '''    
        # output=y_pred
        # target=y_true.long()
        # squares = output ** 2 # (x ^ 2, y ^ 2)
        # loss_1 = ((squares[:, ::2] + squares[:, 1::2]) - 1) ** 2 # (x ^ 2 + y ^ 2 - 1) ** 2

    # # Compute the second loss, 1 - cos
        # loss_2 =  1. - torch.cos(torch.atan2(output[:, 1::2], output[:, ::2]) - target)  
    
        # return self.loss_fn(torch.mean(loss_1 + loss_2))
    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
