from ML_Preparation.Preprocessing_M import *
from ML_Preparation.Feature_Engineering import *

class Predictor:
    
    def __init__(self, 
                 data, feature_engineering_class_object:FeatureEngineering, steps_to_forecast:int,
                 is_forecast:bool = True,
                 rolling_window:str = "3d", sample_time:str = "1d", 
                 roll:bool = True, label:str = "tmp", date_time_column:str = "date_time", get_outliers_out:bool = False):
        

        self.preprocesser = DataPreprocessing(roll = roll, label = label, date_time_column = date_time_column, get_outliers_out = get_outliers_out)
        self.df = self.preprocesser.preprocess_df(data, rolling_window = rolling_window, sample_time = sample_time)

        self.feature_engineerer = feature_engineering_class_object
        scaler = self.feature_engineerer.sc

        if is_forecast == False and label in self.df.columns:
            self.df = self.df.drop(columns = [label], axis = 1)

        # df = df.reindex(columns = self.feature_engineerer.feature_columns)

        self.x = self.feature_engineerer.feature_engineering(only_predict = True, x_for_prediction = self.df, scaler = scaler, steps_to_forecast = steps_to_forecast, skip_scale = True)


    def predict(self, x, model):
        """Given a preprocessed data set containing only features and a given model, make a prediction with the model. Return the predictions with their given room number and date time.
        
        Args:
            x (pandas.DataFrame): feature data which should be already onehotencoded and scaled.
            model: a machine learning model.
        
        Returns:
            pred_df (pandas.DataFrame): a dataset with a prediction in a given room at a given timestamp.
        """

        prefix = "room_number"
        onehot_cols = [col for col in self.feature_engineerer.columns_after_feature_engineering if col.startswith(prefix)]
        
        # if the given model is a Keras model.
        try:
            input_shape = model.input_shape
            pred = model.predict(x.values.reshape(x.shape[0], 1, input_shape[-1]).astype(np.float32))
        except Exception as e:
            pred = model.predict(x)
            
        pred_columns = list()

        if len(pred.shape) > 1:
            for outputs in range(0, pred.shape[1]):
                pred_columns.append(f"prediction_t+{outputs}")
        else:
            pred_columns = ["prediction"]


        pred_df = pd.DataFrame(pred, columns = pred_columns)
        pred_df.index = self.df.index
        pred_df.index.name = "date_time"

        x = pd.DataFrame(x[:, 0, :], columns = self.feature_engineerer.columns_after_feature_engineering)

        # Create a new column by getting the column name for the max value in the one-hot encoded columns
        pred_df[prefix] = x[onehot_cols].idxmax(axis=1).str.replace(f'{prefix}_', '').str.replace("(t-1)", "")

        return pred_df
