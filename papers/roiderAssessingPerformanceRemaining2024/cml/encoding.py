from sklearn.base import TransformerMixin
import pandas as pd
pd.options.mode.chained_assignment = None
from time import time


def get_encoder(method,
                case_id_col=None,
                static_cat_cols=None,
                static_num_cols=None,
                dynamic_cat_cols=None,
                dynamic_num_cols=None,
                fillna=True,
                max_events=None,
                create_dummies=True):
    """
    Get encoder based on the method.

    Parameters
    ----------
    method: str
    case_id_col: str
    static_cat_cols: list
    static_num_cols: list
    dynamic_cat_cols: list
    dynamic_num_cols: list
    fillna: bool
    max_events: int
    create_dummies: bool

    Returns
    -------
    TransformerMixin
        The transformer object.
    """
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols,
                                 fillna=fillna, create_dummies=create_dummies)

    if method == "last":
        return LastStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                    fillna=fillna, create_dummies=create_dummies)

    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                    boolean=False, fillna=fillna, create_dummies=create_dummies)

    elif method == "bool":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                    boolean=True, fillna=fillna, create_dummies=create_dummies)

    elif method == "index":
        return IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                     max_events=max_events, fillna=fillna, create_dummies=create_dummies)

    else:
        return None


class AggregateTransformer(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True, create_dummies=True):
        """
        Constructor.

        Parameters
        ----------
        case_id_col: str
        cat_cols: list
        num_cols: list
        boolean: bool
        fillna: bool
        create_dummies: bool
        """
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.boolean = boolean
        self.fillna = fillna
        self.columns = None
        self.create_dummies = create_dummies

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        TransformerMixin
            Transformer object.

        """
        return self

    def transform(self, X, y=None):
        """
        Transform the data.

        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """

        # Added, otherwise get_dummies does not work
        X[self.cat_cols] = X[self.cat_cols].astype(str)

        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg(["mean", "max", "min", "sum", "std"])
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]

        # transform cat cols
        if self.create_dummies:
            dt_transformed = pd.get_dummies(X[self.cat_cols])
            dt_transformed[self.case_id_col] = X[self.case_id_col]
            del X
            if self.boolean:
                dt_transformed = dt_transformed.groupby(self.case_id_col).max()
            else:
                dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        else:
            dt_transformed = X[self.cat_cols]
            dt_transformed[self.case_id_col] = X[self.case_id_col]

        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric

        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        return dt_transformed


class IndexBasedTransformer(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, max_events=None, fillna=True, create_dummies=True):
        """
        Constructor.

        Parameters
        ----------
        case_id_col: str
        cat_cols: list
        num_cols: list
        max_events: int
        fillna: bool
        create_dummies: bool
        """
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.create_dummies = create_dummies
        self.columns = None

    def fit(self, X, y=None):
        """
        Fit the transformer.
        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        TransformerMixin
            Transformer object.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform the data.

        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        # Added, otherwise get_dummies does not work
        X[self.cat_cols] = X[self.cat_cols].astype(str)

        grouped = X.groupby(self.case_id_col, as_index=False)

        if self.max_events is None:
            self.max_events = grouped.size().max()['size']

        dt_transformed = pd.DataFrame(grouped.apply(lambda x: x.name), columns=[self.case_id_col])
        for i in range(self.max_events):
            dt_index = grouped.nth(i)[[self.case_id_col] + self.cat_cols + self.num_cols]
            dt_index.columns = [self.case_id_col] + ["%s_%s" % (col, i) for col in self.cat_cols] + ["%s_%s" % (col, i)
                                                                                                     for col in
                                                                                                     self.num_cols]
            dt_transformed = pd.merge(dt_transformed, dt_index, on=self.case_id_col, how="left")
        dt_transformed.index = dt_transformed[self.case_id_col]

        # one-hot-encoded cat cols
        if self.create_dummies:
            all_cat_cols = ["%s_%s" % (col, i) for col in self.cat_cols for i in range(self.max_events)]
            dt_transformed = pd.get_dummies(dt_transformed, columns=all_cat_cols).drop(self.case_id_col, axis=1)

        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        return dt_transformed


class LastStateTransformer(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True, create_dummies=True):
        """
        Constructor.

        Parameters
        ----------
        case_id_col: str
        cat_cols: list
        num_cols: list
        fillna: bool
        create_dummies: bool
        """
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.columns = None
        self.create_dummies = create_dummies

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        TransformerMixin
            Transformer object.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform the data.

        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        # Added, otherwise get_dummies does not work
        X[self.cat_cols] = X[self.cat_cols].astype(str)

        dt_last = X.groupby(self.case_id_col).last()

        # transform numeric cols
        dt_transformed = dt_last[self.num_cols]

        # transform cat cols
        if len(self.cat_cols) > 0:
            if self.create_dummies:
                dt_cat = pd.get_dummies(dt_last[self.cat_cols])
            else:
                dt_cat = dt_last[self.cat_cols]
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns

        return dt_transformed


class StaticTransformer(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True, create_dummies=True):
        """
        Constructor.

        Parameters
        ----------
        case_id_col: str
        cat_cols: list
        num_cols: list
        fillna: bool
        create_dummies: bool
        """
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.columns = None
        self.create_dummies = create_dummies

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        TransformerMixin
            Transformer object.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform the data.

        Parameters
        ----------
        X: pd.DataFrame
        y: pd.Series

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        start = time()

        # Added, otherwise get_dummies does not work
        X[self.cat_cols] = X[self.cat_cols].astype(str)

        dt_first = X.groupby(self.case_id_col).first()

        # transform numeric cols
        dt_transformed = dt_first[self.num_cols]

        # transform cat cols
        if len(self.cat_cols) > 0:
            if self.create_dummies:
                dt_cat = pd.get_dummies(dt_first[self.cat_cols])
            else:
                dt_cat = dt_first[self.cat_cols]
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns

        return dt_transformed
