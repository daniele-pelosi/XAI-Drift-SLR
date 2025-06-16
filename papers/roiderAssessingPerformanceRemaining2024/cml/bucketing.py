import numpy as np
from cml import encoding
from sklearn.cluster import KMeans
import pandas as pd


def get_bucketer(method,
                 random_state,
                 encoding_method=None,
                 case_id_col=None,
                 cat_cols=None,
                 num_cols=None,
                 n_clusters=None):
    """
    Get bucketer object.

    Parameters
    ----------
    method: str
        The method to use for bucketing. Possible values: "cluster", "state", "single", "prefix".
    random_state: int
        Random seed.
    encoding_method: str
        The method to use for encoding. Possible values: "laststate", "agg", "index", "combined".
    case_id_col: str
        The name of the column containing the case IDs.
    cat_cols: list
        The list of categorical columns.
    num_cols: list
        The list of numerical columns.
    n_clusters: int
        The number of clusters to use for the "cluster" method.

    Returns
    -------
    object
        Bucket object, provides the following methods:
        - fit
        - predict
        - fit_predict

    """

    if method == "cluster":
        bucket_encoder = encoding.get_encoder(method=encoding_method, case_id_col=case_id_col,
                                              dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        clustering = KMeans(n_clusters, random_state=random_state)
        return ClusterBasedBucketer(encoder=bucket_encoder, clustering=clustering)

    elif method == "state":
        bucket_encoder = encoding.get_encoder(method=encoding_method, case_id_col=case_id_col,
                                              dynamic_cat_cols=cat_cols, dynamic_num_cols=num_cols)
        return StateBasedBucketer(encoder=bucket_encoder)

    elif method == "single":
        return NoBucketer(case_id_col=case_id_col)

    elif method == "prefix":
        return PrefixLengthBucketer(case_id_col=case_id_col)

    else:
        print("Invalid bucketer type")
        return None


class ClusterBasedBucketer(object):

    def __init__(self, encoder, clustering):
        """
        Constructor.
        Parameters
        ----------
        encoder: TransformerMixin
            The encoder to use for encoding the data.
        clustering: object
            Clusting object from sklearn.
        """
        self.encoder = encoder
        self.clustering = clustering

    def fit(self, X, y=None):
        """
        Fit the bucketer.
        Parameters
        ----------
        X: pd.DataFrame
            The data to fit the bucketer on.
        y: pd.Series

        Returns
        -------
        object
            The fitted bucketer.

        """
        dt_encoded = self.encoder.fit_transform(X)

        self.clustering.fit(dt_encoded)

        return self

    def predict(self, X, y=None):
        """
        Predict the bucket for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y: optional

        Returns
        -------
        np.array
            The bucket for each sample.

        """
        dt_encoded = self.encoder.transform(X)

        return self.clustering.predict(dt_encoded)

    def fit_predict(self, X, y=None):
        """
        Fit and predict the bucket for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y: optional

        Returns
        -------
        np.array
            The bucket for each sample.

        """
        self.fit(X)
        return self.predict(X)


class NoBucketer(object):

    def __init__(self, case_id_col):
        """
        Constructor.
        Parameters
        ----------
        case_id_col: str
            The name of the column containing the case IDs.
        """
        self.n_states = 1
        self.case_id_col = case_id_col

    def fit(self, X, y=None):
        """
        Fit the bucketer.
        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        object
            The fitted bucketer.
        """
        return self

    def predict(self, X, y=None):
        """
        Predict the bucket for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y: optional

        Returns
        -------
        np.array
            The bucket for each sample.
        """
        return np.ones(len(X[self.case_id_col].unique()), dtype=np.int32)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


class PrefixLengthBucketer(object):

    def __init__(self, case_id_col):
        """
        Constructor.
        Parameters
        ----------
        case_id_col: str
            The name of the column containing the case IDs.
        """
        self.n_states = 0
        self.case_id_col = case_id_col

    def fit(self, X, y=None):
        """
        Fit the bucketer.
        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        object
            The fitted bucketer.

        """
        sizes = X.groupby(self.case_id_col).size()
        self.n_states = sizes.unique()

        return self

    def predict(self, X, y=None):
        """
        Predict the bucket for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y: optional

        Returns
        -------
        np.array
            The bucket for each sample.
        """
        return X.groupby(self.case_id_col).size().values

    def fit_predict(self, X, y=None):
        """
        Fit and predict the bucket for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y: optional

        Returns
        -------
        np.array
            The bucket for each sample.
        """
        self.fit(X)
        return self.predict(X)


class StateBasedBucketer(object):

    def __init__(self, encoder):
        """
        Constructor.
        Parameters
        ----------
        encoder: TransformerMixin
            The encoder to use for encoding the data.
        """
        self.encoder = encoder
        self.dt_states = None
        self.n_states = 0

    def fit(self, X, y=None):
        """
        Fit the bucketer.
        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        object
            The fitted bucketer.

        """
        dt_encoded = self.encoder.fit_transform(X)

        self.dt_states = dt_encoded.drop_duplicates()
        self.dt_states = self.dt_states.assign(state=range(len(self.dt_states)))

        self.n_states = len(self.dt_states)

        return self

    def predict(self, X, y=None):
        """
        Predict buckets.
        Parameters
        ----------
        X: pd.DataFrame
        y: optional

        Returns
        -------
        np.array
            The bucket for each sample.
        """
        dt_encoded = self.encoder.transform(X)

        dt_transformed = pd.merge(dt_encoded, self.dt_states, how='left')
        dt_transformed.fillna(-1, inplace=True)

        return dt_transformed["state"].astype(int).values

    def fit_predict(self, X, y=None):
        """
        Fit and predict the bucket for each sample.
        Parameters
        ----------
        X: pd.DataFrame
        y: optional

        Returns
        -------
        np.array
            The bucket for each sample.

        """
        self.fit(X)
        return self.predict(X)
