from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class FeatureEngineering:
    """Feature engineering for the use case 'Room classification'. Here, an already preprocessed dataframe is modified further to prepare it for Machine Learning models.
       The taken steps are not supposed to be identical to the feature engineering for the use case 'Prediction of persons in a given room'.
    """

    def __init__(self, df, categorical_features:list = [], label:str = "tmp", automated_feature_engineering:bool = True):

        self.sc = StandardScaler()
        self.df = df
        self.categorical_features = categorical_features
        self.label = label

        try:
            # sort the data by the time frame they were measured at
            self.df = df.sort_values(["date_time"])

            if "date_time" in df.columns:
                self.df = self.df.set_index("date_time")

            # remove features which are not helpful for the ML prediction
            self.df = self.df.drop(columns = ["rssi", "snr", "time_diff_sec"], axis = 1)
            
        except Exception as e:
            print(e)
            pass

        if automated_feature_engineering:
            self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineering()

    
    def feature_engineering(self):
        """Perform feature engineering by calling the other class methods of this class.
        
        Returns:
            :all_X_train (pd.DataFrame): features of the training data.
            :all_X_test (pd.DataFrame): features of the test data.
            :all_y_train (np.Series): labels of the training data.
            :all_y_test (np.Series): labels of the test data.       
        """

        self.df = self.onehotencoding(self.categorical_features)

        x,y = self.split_features_and_labels(self.df, y_col = self.label)

        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split_time_series(x, y)

        self.X_train = self.scale_values(self.X_train, self.sc, test = False)
        self.X_test = self.scale_values(self.X_test, self.sc, test = True)

        return self.X_train, self.X_test, self.y_train, self.y_test

    
    def split_features_and_labels(self, df:pd.DataFrame, y_col:str):
        """Separate featurs and a given label (expects the name of the label).
        
        Args:
            :df (pandas.DataFrame): DataFrame object with the data.
            :y_col (str): name of the label in df.
        
        Returns:
            :x (pd.DataFrame): feature data.
            :y (pd.DataFrame): label data.
        """
        y = df[y_col]
        x = df.drop(columns = [y_col], axis = 1)

        return x,y

    
    def onehotencoding(self, categorical_features:list):
        """Convert categorical features into numerical ones 
        (e.g. color with values 'blue' or 'yellow' leads to two new columns: color_blue, color_yellow. Both with binary values).
        
        Args:
            :categorical_features (list): names of features in the data which include categorical features.
        
        Returns:
            :df (pandas.DataFrame): DataFrame object with onehot encoded features.
        """

        if len(categorical_features) >= 1:
            try:
                for ohe_feature in categorical_features:
                    ohe_df = pd.get_dummies(self.df[f"{ohe_feature}"], prefix = f"{ohe_feature}")
                    # add the new columns to the dataframe
                    self.df = pd.concat([self.df, ohe_df], axis = 1)
                    # drop the old column
                    self.df = self.df.drop(columns = [f"{ohe_feature}"], axis = 1)

                return self.df
                
            except Exception as e:
                print(e)

                return self.df
        
        else:
            return self.df
    

    def scale_values(self, x, scaler, test:bool, non_numerical_features:list = ["hour", "season", "dayofweek", "year", "second", "minute"]):
        """Scale numerical features into the same value range. Ignores non-numerical features and binary encoded columns which are the result of onehotencoding.
        
        Args:
            :x (pd.DataFrame): feature data.
            :scaler (sklearn.preprocessing.StandardScaler): a StandardScaler object.
            :test (bool): information, if x is train or test data.
            :non_numerical_features (list): features which should not be scaled.

        Returns:
            :x_scaled (pd.DataFrame): scaled feature data.
        """
        numerical_features = list()

        for feature in x.columns:
            # ignore binary features or columns which are not int or float
            if x[feature].nunique() > 2 and (np.issubdtype(x[feature].dtype, np.floating) or np.issubdtype(x[feature].dtype, np.integer)):
                if feature not in non_numerical_features:
                    numerical_features.append(feature)

        if test:
            scaled_numerical_features = scaler.transform(x[numerical_features])
        else:
            scaled_numerical_features = scaler.fit_transform(x[numerical_features])

        scaled_numerical_features = pd.DataFrame(scaled_numerical_features, columns = numerical_features)
        scaled_numerical_features = scaled_numerical_features.fillna(0)

        scaled_numerical_features.index = x.index
        # x_scaled keeps the non-numerical columns. Reunify it with the now scaled numerical columns
        x_scaled = x.drop(columns=numerical_features, axis=1)
        x_scaled = pd.concat([x_scaled, scaled_numerical_features], axis=1)

        return x_scaled

    

    def train_test_split_time_series(self, x, y, n_splits = 3, test_size = 0.2):
        """Perform a train or test split using a TimeSeriesSplit class from sklearn. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html.
        
        Args:
            :x (pd.DataFrame): feature data.
            :y (np.Series): label data.
            :n_splits (int): amount of splits 
            :test_size (float): relative amount of the data which is supposed to be test data.

        Returns:
            :X_train (pd.DataFrame): features of the training data.
            :X_test (pd.DataFrame): features of the test data.
            :y_train (np.Series): labels of the training data.
            :y_test (np.Series): labels of the test data.      

        """
        assert x.shape[0] > 100

        try:
            self.ts = TimeSeriesSplit(
                n_splits = n_splits,
                gap = int(x.shape[0] * 0.0001),
                max_train_size = int(x.shape[0] * (1 - test_size) ),
                test_size = int(x.shape[0] * test_size),
            )

            for train_index, test_index in self.ts.split(x):
                X_train, X_test = x.iloc[train_index, :], x.iloc[test_index,:]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            print(e)

            if n_splits > 2:
                return self.train_test_split_time_series(x, y = y, n_splits = int(n_splits - 1), test_size = test_size)
            

    def show_train_test_distribution(self, y_train, y_test):
        """Shows the amount of data points in a train and test data set.
        
        Args:
            :y_train (np.Series): labels of the training data.
            :y_test (np.Series): labels of the test data.

        Returns:
            :tr_te (pd.DataFrame): DataFrame object containing the amounts of train and test data points for each unique label value.
        """

        tr_te = pd.DataFrame([y_train.groupby(y_train.values).count(), y_test.groupby(y_test.values).count()]).T

        tr_te.columns = ["freq_in_train_data", "freq_in_test_data"]

        return tr_te