import numpy as np
import pandas as pd

import warnings

from time import time

from . import univariate

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import Imputer

from .utils.preprocessing import TypeSelector
from .utils.preprocessing import MostFrequentImputer
from .utils.preprocessing import MultiColumnLabelEncoder


class MCPTFeatureEvaluator(BaseEstimator, SelectorMixin):

    def __init__(self, convert_datetime=False,
                 selector_recipe='univariate_unbiased',
                 impute=False,
                 verbose=False,
                 copy=True):
        assert selector_recipe in \
            ['univariate_unbiased',
             'univariate_cscv'], "Provided 'selector_recipe' is not valid!"
        self.convert_datetime = convert_datetime
        self.selector_recipe = selector_recipe
        self.impute = impute
        self.verbose = verbose
        self.copy = copy

    def fit(self, X, y,
            cols_to_discrete=None, method='discrete', measure='mi',
            n_bins_x=5, n_bins_y=5, n_reps=100, cscv_folds=None,
            target='cpu'):
        # make sure n_reps is valid
        assert isinstance(n_reps, int) and n_reps >= 0, \
            "n_reps must be an integer greater than or equal to 0."
        if n_reps == 0:
            warnings.warn("n_reps=0 ... selector_recipe is not applicable.", RuntimeWarning)

        # make sure that a DataFrame has been passed in
        assert isinstance(X, pd.DataFrame)
        # if y is a pandas Series or a numpy array then make
        # sure it is the same length as the DataFrame
        if isinstance(y, pd.Series) or isinstance(y, np.ndarray):
            assert len(y) == X.shape[0]
            if isinstance(y, np.ndarray):
                # if it is a numpy array then need to make it into a series
                # and assign a name to that series
                y_name = 'y_auto'
                y = pd.Series(y, name='y_auto')
            else:
                # capture the name from the y pandas Series
                y_name = y.name
            # Capture original predictor columns prior to adding y into X
            self.original_predictor_columns = X.columns
            # add y into X so that it can be pre-processed correctly
            X = X.assign(y_auto=y).rename(columns={'y_auto': y_name})
        else:
            # make sure that 'y' is a column in X
            assert y in X.columns, "The target column 'y' does not exist in 'X'!"
            y_name = y
            # Capture the original predictor columns without removing the target
            self.original_predictor_columns = X.columns

        # Convenience check:
        # Remove any columns that might have a single value as information
        # calculations will not really be accurate
        for col in X.columns:
            if X[col].nunique() == 1:
                if col == y_name:
                    raise RuntimeError("The target variable has an insufficient " +
                                      "number of unique values!")
                X = X.drop([col], axis=1)
                warnings.warn("Removing column '" + col +
                              "' due to non-unique values", RuntimeWarning)

        # convert any columns to discrete that are indicated in the fit method
        if cols_to_discrete is not None:
            try:
                X[cols_to_discrete] = X[cols_to_discrete].astype(object)
            except KeyError:
                cols_error = list(set(cols_to_discrete) - set(X.columns))
                raise KeyError("The DataFrame does not " +
                               "include the following " +
                               "columns to discretize: %s" % cols_error)

        t0 = time()
        # Capture the column names in the DataFrame. These will be used
        # later in the presentation of the results
        self.numeric_cols = TypeSelector(np.number, True).fit_transform(X)
        self.bool_cols = TypeSelector("bool", True).fit_transform(X)
        self.category_cols = TypeSelector("category", True).fit_transform(X)
        self.object_cols = TypeSelector("object", True).fit_transform(X)
        self.datetime_cols = TypeSelector("datetime", True).fit_transform(X)
        self.preprocessed_columns = []
        self.column_type_mask = []

        if self.verbose:
            t1 = time()
            print("Type selector time: {0:.3g} sec".format(t1 - t0))

        t2 = time()
        # Generate the feature union with all the possible different dtypes
        if self.impute:
            self.preprocess_features = \
                FeatureUnion(transformer_list=[
                    ("numeric_features", make_pipeline(
                        TypeSelector(np.number),
                        Imputer(strategy="median")
                    )),
                    ("boolean_features", make_pipeline(
                        TypeSelector("bool"),
                        Imputer(strategy="most_frequent")
                    )),
                    ("categorical_features", make_pipeline(
                        TypeSelector("category"),
                        MostFrequentImputer(),
                        MultiColumnLabelEncoder(self.category_cols)
                    )),
                    ("object_features", make_pipeline(
                        TypeSelector("object"),
                        MostFrequentImputer(),
                        MultiColumnLabelEncoder(self.object_cols)
                    )),
                    ("datetime_features", make_pipeline(
                        TypeSelector("datetime"),
                        Imputer(strategy="most_frequent")
                    ))
                ])
        else:
            # don't impute missing values
            # less execute time
            self.preprocess_features = \
                FeatureUnion(transformer_list=[
                    ("numeric_features", make_pipeline(
                        TypeSelector(np.number)
                    )),
                    ("boolean_features", make_pipeline(
                        TypeSelector("bool")
                    )),
                    ("categorical_features", make_pipeline(
                        TypeSelector("category"),
                        MultiColumnLabelEncoder(self.category_cols)
                    )),
                    ("object_features", make_pipeline(
                        TypeSelector("object"),
                        MultiColumnLabelEncoder(self.object_cols)
                    )),
                    ("datetime_features", make_pipeline(
                        TypeSelector("datetime")
                    ))
                ])

        # If some of the dtypes are not present in the data set
        # and the feature union is run as-is, an error will be thrown.
        # As a result, those feature pipelines without any dtypes
        # need to be set to None prior to the FeatureUnion being executed
        # If they are present, then they need to be added to the
        # list of all columns being evaluated
        #
        # In addition, the categorical/object columns need to be tracked
        # so that the proper MI algorithm can be used in the calculation
        # (cont/cont, discrete/cont, discrete/discrete)
        #
        # Numerics
        if len(self.numeric_cols) == 0:
            self.preprocess_features.set_params(numeric_features=None)
        else:
            self.preprocessed_columns += list(self.numeric_cols)
            for i in range(len(self.numeric_cols)):
                self.column_type_mask += ['numeric']
        # Booleans
        if len(self.bool_cols) == 0:
            self.preprocess_features.set_params(boolean_features=None)
        else:
            self.preprocessed_columns += list(self.bool_cols)
            for i in range(len(self.bool_cols)):
                self.column_type_mask += ['discrete']
        # Categorical
        if len(self.category_cols) == 0:
            self.preprocess_features.set_params(categorical_features=None)
        else:
            self.preprocessed_columns += list(self.category_cols)
            for i in range(len(self.category_cols)):
                self.column_type_mask += ['discrete']
        # Object
        if len(self.object_cols) == 0:
            self.preprocess_features.set_params(object_features=None)
        else:
            self.preprocessed_columns += list(self.object_cols)
            for i in range(len(self.object_cols)):
                self.column_type_mask += ['discrete']
        # Datetime
        if len(self.datetime_cols) == 0:
            self.preprocess_features.set_params(datetime_features=None)
        else:
            self.preprocessed_columns += list(self.datetime_cols)
            for i in range(len(self.datetime_cols)):
                self.column_type_mask += ['datetime']

        X = self.preprocess_features.fit_transform(X)

        if self.verbose:
            t3 = time()
            print("Preprocess time: {0:.3g} sec".format(t3 - t2))

        # Convert back to a DataFrame so we can properly extract the target
        # variable. There may be a better way to do this but wanted to preprocess
        # the target variable inline with all the other variables
        X = pd.DataFrame(data=X, columns=self.preprocessed_columns)
        # extract y series from X after preprocessing, determine if it is
        # discrete or continuous, and remove from X
        y = X[y_name]
        X = X.drop([y_name], axis=1)

        # Derive 'target_is_discrete'
        target_is_discrete = (y_name in self.category_cols) or \
                                (y_name in self.object_cols)
        # Adjust 'self.column_type_mask' and 'self.columns' given that
        # 'y' was just separated out from 'X'
        self.preprocessed_columns = np.array(self.preprocessed_columns)
        self.column_type_mask = np.array(self.column_type_mask)
        predictor_idx = np.where(self.preprocessed_columns != y_name)[0]
        self.column_type_mask = self.column_type_mask[predictor_idx]
        self.preprocessed_columns = self.preprocessed_columns[predictor_idx]

        # Next steps ... run the MCPT
        self.method = method
        self.measure = measure
        self.target = target
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.n_mcpt_reps = n_reps
        if cscv_folds is None and ('cscv' in self.selector_recipe):
            self.n_cscv_folds = 4
            warnings.warn("No value provided for 'cscv_folds'. '" +
                          "Setting to '4' CSCV folds by default.", RuntimeWarning)
        else:
            self.n_cscv_folds = cscv_folds

        kwargs = {'method': self.method, 'measure': self.measure, \
                  'target_is_discrete': target_is_discrete, \
                  'column_type_mask': self.column_type_mask, \
                  'n_bins_x': self.n_bins_x, 'n_bins_y': self.n_bins_y, \
                  'n_reps': self.n_mcpt_reps, 'cscv_folds': self.n_cscv_folds, \
                  'target': self.target, 'verbose': self.verbose}

        if self.selector_recipe in ['univariate_unbiased','univariate_cscv']:
            info_matrix = univariate.screen_univariate(X.values, y.values, **kwargs)
        else:
            raise ValueError

        var_series = pd.Series(self.preprocessed_columns, name='Variables')
        if self.measure == 'mi':
            measure_name = 'MI'
        else:
            measure_name = 'UR'
        if self.n_mcpt_reps > 0:
            col_names = [measure_name, 'Solo p-value', 'Unbiased p-value']
        else:
            col_names = [measure_name]
        if self.n_cscv_folds is not None:
            col_names += ['P(<=median)']

        self.information = pd.DataFrame(info_matrix, columns=col_names)
        self.information.insert( 0, 'Variable', var_series)

        self.information = self.information.sort_values(by=measure_name, ascending=False)
        self.information = self.information.reset_index(drop=True)

        return self


    def _get_support_mask(self):
        check_is_fitted(self, 'information')

        orig_cols = self.original_predictor_columns

        if self.n_mcpt_reps > 0:
            if self.selector_recipe == "univariate_unbiased":
                # Get the variable names
                variable_mask = self.information['Unbiased p-value'] < 0.05
                selected = self.information['Variable'][variable_mask].values
            elif self.selector_recipe == "univariate_cscv":
                variable_mask_1 = self.information['P(<=median)'] <= 0.2
                variable_mask_2 = self.information['Unbiased p-value'] < 0.05
                variable_mask = variable_mask_1.values & variable_mask_2.values
                selected = self.information['Variable'][variable_mask].values
            else:
                raise ValueError
            mask = np.array([True if col in selected else False for col in orig_cols])
        else:
            # self.n_mcpt_reps = 0 .. return all original columns
            mask = np.repeat(True, len(orig_cols))

        return mask
