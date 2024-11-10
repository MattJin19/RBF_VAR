from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gosdt.model.threshold_guess import compute_thresholds
from treefarms import TREEFARMS
from treefarms.model.tree_classifier import TreeClassifier

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

DEFAULT_TF_CONFIG = {
    "depth_budget": 4,
    "rashomon_bound_adder": 0.03,
    "rashomon_bound_multiplier": 0,
    "rashomon_bound": 0,
    "rashomon_ignore_trivial_extensions": True,
    "regularization": 0.02,
    "verbose": False
}
#================================================================

class TreeClassifierWrapper(TreeClassifier):
    def predict(self, X):
        """
        A faster version of the predict function from TreeClassifier

        Requires
        ---
        the set of features used should be pre-encoding if an encoder is used

        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the prediction associated with each row
        """
        if not self.encoder is None: # Perform an encoding if an encoding unit is specified
            X = pd.DataFrame(self.encoder.encode(X.values[:,:]), columns=self.encoder.headers)
        
        predictions = []
        (n, m) = X.shape

        # By only accessing .values once, we avoid duplicating the
        # entire array a ton of times
        data = X.values
        for i in range(n):
            prediction, _ = self.classify(data[i,:])
            predictions.append(prediction)
        return np.array(predictions)

def json_set_to_model_list(
    json_set: set,
    X_train: pd.DataFrame = None,
    y_train: pd.DataFrame = None,
) -> list:
    '''
    Converts a set of JSON representations of TreeClassifiers
    to a list of TreeClassifiers that may be easily iterated over
    Args:
        json_set : set -- a set of trees represented as json strings
        X_train : pd.DataFrame -- the X values used to fit these trees.
            If provided, these are used to populate the loss attribute of
            each tree
        y_train : pd.DataFrame -- the y values used to fit these trees.
            If provided, these are used to populate the loss attribute of
            each tree
    '''
    json_list = list(json_set)
    model_list = []
    if X_train is not None and y_train is not None:
        model_list = [
            TreeClassifierWrapper(json.loads(cur_dict), X=X_train, y=pd.DataFrame(y_train)) for cur_dict in json_list
        ]
    else:
        model_list = [TreeClassifierWrapper(json.loads(cur_dict)) for cur_dict in json_list]
    return model_list


class TreeFarmsWrapper:

    def __init__(self, tf_config: dict = DEFAULT_TF_CONFIG):
        self.configuration: dict = tf_config
        self.tf: TREEFARMS = TREEFARMS(self.configuration)
        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.X_train_no_nan_cols: pd.DataFrame = None
        self.observed_labels: list[bool] = None
        self.model_set_as_set: set[str] = set()
        self.model_set_as_list: list[TreeClassifierWrapper] = list()
        self.unusable_data_prediction = None
        self.fit_on_unusable_data: bool = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        '''
        If there are any columns with missing values, we need to drop them to use treefarms.
        If that means dropping all columns, we handle this with a flag.
        We also handle the standing bug with treefarms where, if all rows are equivalent,
        TreeFarms crashes, in that same case.
        '''
        self.X_train_no_nan_cols = self.X_train.dropna(axis=1)
        self.observed_labels = list(self.y_train.unique().astype(bool))
        if self.X_train_no_nan_cols.shape[1] == 0 or (self.X_train_no_nan_cols.nunique(axis=0) == 1).all():
            self.unusable_data_prediction = self.y_train.mode()[0]
            self.fit_on_unusable_data = True
        else:
            self.tf.fit(self.X_train_no_nan_cols, self.y_train)
            '''
            Getting a python set version of the model set.
            We use the JSON representation of each tree to support 
            equality comparisons for set intersect/union, as well
            as to allow for corrections on edge cases treefarms 
            does not handle well (i.e. single label datasets)
            or to remap features back to the original dataset, 
            for the processing we did earlier when dropping nan columns
            '''
            self.model_set_as_set = {self._get_corrected_json(i) for i in range(self.get_tree_count())}
            self.model_set_as_list = json_set_to_model_list(self.model_set_as_set, self.X_train, self.y_train)
            self.fit_on_unusable_data = False

    def __len__(self):
        return len(self.model_set_as_set)

    def _get_corrected_json(self, index: int):
        '''
        Simple wrapper function to make sure our string tree representations
        are consistent and always predict the relevant label.
        '''
        json_dict = json.loads(self.tf[index].json())

        if len(self.observed_labels) == 1:
            self._correct_predicted_label(json_dict, self.observed_labels[0])

        if (self.X_train_no_nan_cols.shape[1] < self.X_train.shape[1] and self.X_train_no_nan_cols.shape[1] > 0):
            self._correct_feature_indices(json_dict)

        return json.dumps(json_dict)

    def _correct_feature_indices(self, tree_rep: dict):
        '''
        Simple wrapper function to make sure our string tree representations
        use indices relevant ot the full dataset.

        Args: 
            tree_rep : dict -- the JSON representation of a tree

        Modifies: 
            tree_rep : dict -- the JSON representation of a tree
            (Modified to use the correct feature indices for the full dataset, 
             rather than the subset of columns it was trained on from X_train_no_nan_cols)
        '''
        if 'feature' in tree_rep:
            tree_rep['feature'] = self.X_train.columns.get_loc(self.X_train_no_nan_cols.columns[tree_rep['feature']])
            self._correct_feature_indices(tree_rep['true'])
            self._correct_feature_indices(tree_rep['false'])

    def _correct_predicted_label(self, json_dict: dict, target_label: bool):
        '''
        Simple wrapper function to make sure our string tree representations
        are consistent and always predict the relevant label.
        '''
        if 'prediction' in json_dict:
            json_dict['prediction'] = int(target_label)
        else:
            self._correct_predicted_label(json_dict['true'], target_label)
            self._correct_predicted_label(json_dict['false'], target_label)

    def get_tree_count(self) -> int:
        return self.tf.get_tree_count()

    def __getitem__(self, index: int) -> TreeClassifierWrapper:
        return self.model_set_as_list[index]
#================================================================

def get_mr(X, y, predictor, metric=accuracy_score):
    res = {
        'var': [],
        'mr': [],
        'mr_og_r2': [],
        'mr_pert_r2': []
    }
    og_r2 = metric(y, predictor(X))
    for col in X.columns:
        imputed_df = X.copy()
        imputed_df[col] = X[col].sample(frac=1).values
        pert_r2 = metric(y, predictor(imputed_df))

        res['var'] = res['var'] + [col]
        res['mr_og_r2'] = res['mr_og_r2'] + [og_r2]
        res['mr_pert_r2'] = res['mr_pert_r2'] + [pert_r2]
        res['mr'] = res['mr'] + [og_r2 - pert_r2]
    return res

def binarize_with_GOSDT_guesses(X: pd.DataFrame,
                                y: pd.Series,
                                n_est: int = 40,
                                max_depth: int = 1) -> tuple[pd.DataFrame, list, pd.Index]:
    X_guessed, thresholds, header, _ = compute_thresholds(X, y, n_est, max_depth)
    return X_guessed, thresholds, header

def nansafe_cut(X, ts):
    df = X.copy()
    colnames = X.columns
    for j in range(len(ts)):
        for s in range(len(ts[j])):
            X[colnames[j]+'<='+str(ts[j][s])] = 1
            # The following line is the only change from 
            # GOSDT's cut function
            k = (df[colnames[j]] > ts[j][s]) | (df[colnames[j]].isna())
            X.loc[k, colnames[j]+'<='+str(ts[j][s])] = 0
        X = X.drop(colnames[j], axis=1)
    return X

if __name__ == '__main__':
    # Read and binarize our data ===================
    target_data = "./data/fico_full.csv"
    save_file = 'fico_treefarms_stats.csv'

    df = pd.read_csv(target_data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    _, thresholds, headers = binarize_with_GOSDT_guesses(X_train, y_train)
    X_train = nansafe_cut(X_train, thresholds)[headers]
    X_test = nansafe_cut(X_test, thresholds)[headers]
    
    # Construct our Rashomon set ===================
    tf = TreeFarmsWrapper(DEFAULT_TF_CONFIG)

    tf.fit(
        X_train,
        y_train
    )

    train_accs = []
    train_f1s = []
    test_accs_complete = []
    test_f1s_complete = []
    test_accs_all = []
    test_f1s_all = []

    train_losses = []
    n_leaves = []

    all_results = {}
    for model_ind in tqdm(range(tf.get_tree_count())):
        model = tf[model_ind]
        preds_train = model.predict(X_train)

        train_losses.append(
            (preds_train != y_train).mean() + model.leaves() * DEFAULT_TF_CONFIG['regularization']
        )
        n_leaves.append(model.leaves())
        train_accs.append(accuracy_score(y_train, preds_train))
        train_f1s.append(f1_score(y_train, preds_train))

        X_test_complete = X_test.dropna(axis=0)
        y_test_complete = y_test[X_test_complete.index]
        preds_test_complete = model.predict(X_test_complete)
        test_accs_complete.append(accuracy_score(y_test_complete, preds_test_complete))
        test_f1s_complete.append(f1_score(y_test_complete, preds_test_complete))

        mr_dict = get_mr(X_train, y_train, model.predict)

        for i in range(len(mr_dict['var'])):
            if f"{mr_dict['var'][i]}_importance" in all_results:
                all_results[f"{mr_dict['var'][i]}_importance"] = all_results[f"{mr_dict['var'][i]}_importance"] + [mr_dict['mr'][i]]
            else:
                all_results[f"{mr_dict['var'][i]}_importance"] = [mr_dict['mr'][i]]

    all_results['train_acc'] = train_accs
    all_results['train_f1'] = train_f1s
    all_results['test_acc_complete'] = test_accs_complete
    all_results['test_f1_complete'] = test_f1s_complete
    # all_results['test_acc_all'] = test_accs_all
    # all_results['test_f1_all'] = test_f1s_all
    all_results['n_leaves'] = n_leaves
    all_results['train_loss'] = train_losses

    all_results = pd.DataFrame(all_results)
    all_results.to_csv(save_file, index=False)
