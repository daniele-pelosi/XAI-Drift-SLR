import optuna
from cml import train
import joblib
import pandas as pd


class Objective:
    """
    Objective that defines the procedure for one optuna trial.
    """
    def __init__(self,
                 search_space,
                 bucketing_method,
                 encoding_method,
                 input_data_location,
                 output_location,
                 seed,
                 model_args,
                 cml_method):
        """
        Constructor for the Objective class.

        Parameters
        ----------
        search_space: dict
            Search space to sample hyperparmeters. Keys are the hyperparameter names and values are the possible values.
        bucketing_method: str
            The bucketing method to use. Possible values are 'prefix', 'state', 'single', and 'cluster'.
        encoding_method: str
            The encoding method to use. Possible values are 'laststate', 'agg', 'index', and 'combined'.
        input_data_location: str
            Folder with input data.
        output_location: str
            Folder to save the output.
        seed: int:
            Random seed.
        model_args: dict
            Arguments that will be passedon to the model.
        cml_method: RegressorMixin
            The method to use.
        """
        self.search_space = search_space
        self.seed = seed
        self.input_data_location = input_data_location
        self.bucketing_method = bucketing_method
        self.encoding_method = encoding_method
        self.cml_method = cml_method
        self.model_args = model_args
        self.output_location = output_location

    def __call__(self,
                 trial: optuna.trial.Trial):
        """
        This function is called by optuna to evaluate the objective function.

        Parameters
        ----------
        trial: optuna.trial.Trial
            The optuna trial object that is used to sample hyperparameters.

        Returns
        -------
        float
            The mean absolute error of the predictions.
        """

        self.model_args['n_estimators'] = trial.suggest_categorical('n_estimators', self.search_space['n_estimators'])
        self.model_args['learning_rate'] = trial.suggest_categorical('learning_rate', self.search_space['learning_rate'])
        self.model_args['subsample'] = trial.suggest_categorical('subsample', self.search_space['subsample'])
        self.model_args['colsample_bytree'] = trial.suggest_categorical('colsample_bytree', self.search_space['colsample_bytree'])
        self.model_args['max_depth'] = trial.suggest_categorical('max_depth', self.search_space['max_depth'])

        config = joblib.load(self.input_data_location + '/config.pkl')

        encoding_dict = {  # cls_encoding
            "laststate": ["static", "last"],
            "agg": ["static", "agg"],
            "index": ["static", "index"],
            "combined": ["static", "last", "agg"]}

        config['dynamic_categorical_columns'].append(config['activity_column'])

        encoding_args = {'case_id_col': config['case_id_column'],
                         'static_cat_cols': config['static_categorical_columns'],
                         'static_num_cols': config['static_numerical_columns'],
                         'dynamic_cat_cols': config['dynamic_categorical_columns'],
                         'dynamic_num_cols': config['dynamic_numerical_columns'],
                         'fillna': True}

        encoding_methods = encoding_dict[self.encoding_method]

        train_df = pd.read_pickle(self.input_data_location + '/train.pkl')
        val_df = pd.read_pickle(self.input_data_location + '/val.pkl')

        train_df = train_df[train_df['remaining_time'] != 0]
        val_df = val_df[val_df['remaining_time'] != 0]

        train_df[config['timestamp_column']] = pd.to_datetime(train_df[config['timestamp_column']], format='mixed',
                                                              infer_datetime_format=True)
        train_df = train_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

        val_df[config['timestamp_column']] = pd.to_datetime(val_df[config['timestamp_column']], format='mixed',
                                                            infer_datetime_format=True)
        val_df = val_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

        preds_df = train.train(train_df,
                               val_df,
                               config=config,
                               bucketing_method=self.bucketing_method,
                               # n_clusters=n_clusters,
                               random_state=self.seed,
                               encoding_methods=encoding_methods,
                               encoding_args=encoding_args,
                               cls_method=self.cml_method,
                               cls_args=self.model_args)

        med = preds_df['labels'].median()
        med_loss = (med - preds_df['labels']).abs().mean()
        val_nmae = (preds_df['labels'] - preds_df['preds']).abs().mean() / med_loss

        trial.set_user_attr("best_val_nmae", val_nmae)

        return (preds_df['labels'] - preds_df['preds']).abs().mean()
