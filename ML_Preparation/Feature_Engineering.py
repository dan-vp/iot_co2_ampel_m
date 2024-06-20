from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class FeatureEngineering:
    """Feature engineering for the use case 'Room classification'. Here, an already preprocessed dataframe is modified further to prepare it for Machine Learning models.
       The taken steps are not supposed to be identical to the feature engineering for the use case 'Prediction of persons in a given room'.
    """

    def __init__(self, df, categorical_features:list = [], label:str = "tmp", 
                 skip_scale:bool = False,
                 automated_feature_engineering:bool = True):

        self.sc = StandardScaler()
        self.df = df
        self.categorical_features = categorical_features
        self.label = label

        try:
            # sort the data by the time frame they were measured at
            self.df = df.sort_values(["date_time"])

            if "date_time" in df.columns:
                self.df = self.df.set_index("date_time")
            
        except Exception as e:
            print(e)
            pass

        if automated_feature_engineering:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.feature_engineering(skip_scale = skip_scale)

    
    def feature_engineering(self, steps_to_forecast:int, input_time_steps:int = 1, skip_scale:bool = False, scaler = None, x_for_prediction:pd.DataFrame = None, only_predict:bool = False):
        """Perform feature engineering by calling the other class methods of this class. Can be used for model training and simple predictions of data points.

        Args:
            steps_to_forecast (bool): amount of time steps to create as a label (value 2 --> two timesteps of a label value will be created --> two labels to predict)
            input_time_steps (int): amount of time steps to use for feature data as an input. Default value set to 1.
            only_predict (bool): perform no train-test-split if True. Default value set to False.
            skip_scale (bool): do not scale data if True. Default value set to False.
            scaler: scaler to use for scaling the values. If the input is invalid, the scaler of this class object will be used instead.
            x_for_prediction (pd.DataFrame): feature data to predict during the deployment phase.
            only_predict (bool): True to avoid any train-test-splits or feature-label splits. This should be set to True for data without a known label.
        
        Returns:
            if only_predict is set to False:
                :all_X_train (np.array): features of the training data.
                :all_X_test (np.array): features of the test data.
                :all_y_train (np.array): labels of the training data.
                :all_y_test (np.array): labels of the test data.

            if only_predict is set to True:
                :data_array (np.array): scaled features of the data shaped into a 3D-numpy array.
        """

        if scaler == None:
            scaler = self.sc

        if only_predict == False:
            self.df = self.df.sort_index(axis=1)
            self.df = self.onehotencoding(self.categorical_features)

            # for the model training
            x,y = self.split_features_and_labels(self.df, y_col = self.label)
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split_time_series(x, y)

            val_index = int(0.1 * self.X_train.shape[0])
            self.X_val = self.X_train.iloc[-val_index:, :]
            self.y_val = self.y_train.iloc[-val_index:]
            self.X_train = self.X_train.iloc[:-val_index, :]
            self.y_train = self.y_train.iloc[:-val_index]


            if skip_scale == False:
                try:
                    self.X_train = self.scale_values(self.X_train, scaler, test = False)
                    self.X_val = self.scale_values(self.X_val, scaler, test = True)
                    self.X_test = self.scale_values(self.X_test, scaler, test = True)
                except:
                    print("The scaler which the user has given as an input is invalid. Using scaler of this class object instead.")
                    self.X_train = self.scale_values(self.X_train, self.sc, test = False)
                    self.X_val = self.scale_values(self.X_val, self.sc, test = True)
                    self.X_test = self.scale_values(self.X_test, self.sc, test = True)

            df_train = self.X_train
            df_train[self.label] = self.y_train

            df_val = self.X_val
            df_val[self.label] = self.y_val

            df_test = self.X_test
            df_test[self.label] = self.y_test

            train_reframed = self.transform_data_for_forecasting(df_train, self.label, input_time_steps, steps_to_forecast)
            val_reframed = self.transform_data_for_forecasting(df_val, self.label, input_time_steps, steps_to_forecast)
            test_reframed = self.transform_data_for_forecasting(df_test, self.label, input_time_steps, steps_to_forecast)

            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.transform_to_numpy_array(train_reframed, val_reframed, test_reframed, steps_to_forecast)
            
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        
        else:
            # for the deployment (or simple predictions of data with unknown labels)
            self.df = x_for_prediction.copy()
            self.df = self.onehotencoding(self.categorical_features)
            
            if skip_scale == False:
                try:
                    self.df = self.scale_values(self.df, scaler, test = True)
                except Exception as e:
                    print("The scaler which the user has given as an input is invalid. Using scaler of this class object instead.")
                    self.df = self.scale_values(self.df, self.sc, test = True)

            if "date_time" in self.df.columns:
                self.df = self.df.set_index("date_time")

            self.df = self.transform_data_for_forecasting_without_label(self.df, input_time_steps)

            data_array = self.transform_to_numpy_array_without_label(self.df)

            return data_array

    
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
                    ohe_df = pd.get_dummies(self.df[f"{ohe_feature}"], prefix = f"{ohe_feature}", dtype = "int")
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
    

    def scale_values(self, x, scaler, test:bool):
        """Scale numerical features into the same value range. Ignores non-numerical features and binary encoded columns which are the result of onehotencoding.
        
        Args:
            :x (pd.DataFrame): feature data.
            :scaler (sklearn.preprocessing.StandardScaler): a StandardScaler object.
            :test (bool): information, if x is train or test data.

        Returns:
            :x_scaled (pd.DataFrame): scaled feature data.
        """

        if test:
            scaled_numerical_features = scaler.transform(x[self.feature_order_for_scaler])
        else:
            numerical_features = list()

            for feature in x.columns:
                # ignore binary features or columns which are not int or float
                if (pd.api.types.is_integer_dtype(x[feature])) or (pd.api.types.is_float_dtype(x[feature])):
                    if set(x[feature].unique()).issubset({0, 1}) == False:
                        numerical_features.append(feature)
                        
            self.feature_order_for_scaler = numerical_features
            scaled_numerical_features = scaler.fit_transform(x[self.feature_order_for_scaler])

        scaled_numerical_features = pd.DataFrame(scaled_numerical_features, columns = self.feature_order_for_scaler)
        scaled_numerical_features = scaled_numerical_features.fillna(0)

        scaled_numerical_features.index = x.index
        # x_scaled keeps the non-numerical columns. Reunify it with the now scaled numerical columns
        x_to_be_scaled = x.drop(columns=self.feature_order_for_scaler, axis=1)
        x_scaled = pd.concat([x_to_be_scaled, scaled_numerical_features], axis=1)

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
    

    def transform_data_for_forecasting(self, data, label_name, input_time_steps, steps_to_forecast, dropna = True):

        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(input_time_steps, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col, i)) for col in data.columns]
            
        self.columns_after_feature_engineering = names

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, steps_to_forecast):
            cols.append(data[[f"{label_name}"]].shift(-i))
            if i == 0:
                names += [f"{label_name}(t)"]
            else:
                names += [f"{label_name}(t+{i})"]
            # put it all together
            data_reframed = pd.concat(cols, axis=1)
            data_reframed.columns = names
            # drop rows with NaN values
            if dropna:
                data_reframed.dropna(inplace=True)

        return data_reframed
    

    def transform_data_for_forecasting_without_label(self, data, input_time_steps):

        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(input_time_steps, 0, -1):
            cols.append(data.shift(i))
            names += [('%s(t-%d)' % (col, i)) for col in data.columns]

        data.columns = names

        self.columns_after_feature_engineering = names

        return data
    


    def transform_to_numpy_array(self, train_data, val_data, test_data, steps_to_forecast):
        # split into train and test sets

        train_X, train_y = train_data.iloc[:, :-steps_to_forecast], train_data.iloc[:, -steps_to_forecast:]
        val_X, val_y = val_data.iloc[:, :-steps_to_forecast], val_data.iloc[:, -steps_to_forecast:]
        test_X, test_y = test_data.iloc[:, :-steps_to_forecast], test_data.iloc[:, -steps_to_forecast:]

        train_X = self.scale_values(x = train_X, test = False, scaler = self.sc)
        val_X = self.scale_values(x = val_X, test = True, scaler = self.sc)
        test_X = self.scale_values(x = test_X, test = True, scaler = self.sc)

        train_X = train_X.values
        val_X = val_X.values
        test_X = test_X.values

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        train_X = np.asarray(train_X).astype('float32')
        train_y = np.asarray(train_y).astype('float32')

        val_X = np.asarray(val_X).astype('float32')
        val_y = np.asarray(val_y).astype('float32')

        test_X = np.asarray(test_X).astype('float32')
        test_y = np.asarray(test_y).astype('float32')

        return train_X, val_X, test_X, train_y, val_y, test_y
    

    def transform_to_numpy_array_without_label(self, data):
        # split into train and test sets

        data_scaled = self.scale_values(x = data, test = True, scaler = self.sc)

        data_scaled = data_scaled.values

        # reshape input to be 3D [samples, timesteps, features]
        data_shaped = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

        data_array = np.asarray(data_shaped).astype('float32')

        return data_array