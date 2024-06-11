import os, io
import pandas as pd
import tarfile, zipfile
import numpy as np
from sklearn.ensemble import IsolationForest
pd.options.mode.chained_assignment = None  # default='warn'



class DataExtractor:
    """Extracts the historic CO2-Ampeldatensatz and converts the files into a single dataframe."""

    def __init__(self, first_directory, new_directory):
        self.first_directory = first_directory
        self.new_directory = new_directory


    def create_df(self):
        """Create a pandas.DataFrame object by extracting zip files and/ or reading .dat files."""
        self.extract_zip_files(self.first_directory, self.new_directory)
        self.delete_zip_files(self.first_directory)
        self.df = self.get_data(self.new_directory)

        return self.df


    def extract_zip_files(self, directory, new_directory):
        """Extract all zip and tar files in the parameter directory and its subdirectories.
        
        Args:
            :directory (str): name of the current directory with all the zip files.
            :new_directory (str): name of the directory where the files are supposed to be extracted to.
        """
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(new_directory)
                            print(f"Extracted {file_name} in {new_directory}")
                    except zipfile.BadZipFile as e:
                        print(f"Failed to extract {file_name}: {e}")
                elif file_name.endswith(".tar") or file_name.endswith(".tar.gz") or file_name.endswith(".tgz") or file_name.endswith(".tar.bz2"):
                    try:
                        with tarfile.open(file_path, 'r') as tar_ref:
                            tar_ref.extractall(new_directory)
                            print(f"Extracted {file_name} in {new_directory}")
                    except tarfile.TarError as e:
                        print(f"Failed to extract {file_name}: {e}")


    def delete_zip_files(self, directory):
        """Delete all zip and tar files in a directory and its subdirectories.
        
        Args:
            :directory (str): name of the directory where the zip files are supposed to be removed.
        """
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")):
                    try:
                        os.remove(file_path)
                        print(f"File {file_name} has been deleted.")
                    except Exception as e:
                        print(f"Failed to delete {file_name}: {e}")


    def get_data(self, directory):
        """Read all extracted .dat files into a pandas DataFrame.
        
        Args:
            :directory (str): name of the directory to extract the .dat files from.

        Returns:
            :df (pandas.DataFrame): a DataFrame object containing the read data.
        """
        dataframes = []
        for root, _, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    # Read the .dat file into a DataFrame
                    df = pd.read_csv(file_path, delimiter=';', 
                                    header = 1, encoding='unicode_escape', on_bad_lines='skip')
                    dataframes.append(df)
                except Exception as e:
                    print(f"Failed to read {file_name} into DataFrame: {e}")
                    continue

        # Concatenate all DataFrames into a single DataFrame
        if dataframes:
            print("Read data successfully.")
            final_df = pd.concat(dataframes, ignore_index=True)
            print(f"Data contains {final_df.shape[0]} data points and {final_df.shape[1]} columns.")
            return final_df
        else:
            print(f"No .dat files found in {self.new_directory}. \n Trying to extract files from the original directory {self.first_directory}")
            # in case there were no files to extract and no new directory has been created, try reading the data from the original first directory.
            try:
                return self.get_data(self.first_directory)
            except:
                print(f"No .dat files found in {self.first_directory}. Empty DataFrame returned.")
                return pd.DataFrame()
    
    

class DataPreprocessing:
    """Performs data preprocessing on the CO2-Ampeldaten."""

    def __init__(self, get_outliers_out = True, roll:bool = True, label:str = "tmp", date_time_column:str = "date_time"):
        """
        Args
            :get_outliers_out (bool): remove outliers if True, otherwise don't.
            :roll (bool): create rolling windows on the data if True, otherwise don't.
            :label (str): name of the label in the data.
            :date_time_column (str): the column name in the data which contains the date time information.
        """
        self.get_outliers_out = get_outliers_out
        self.roll = roll
        self.label = label
        self.date_time_column = date_time_column


    def preprocess_df(self, df, rolling_window:str, sample_time:str = "15min"):
        """Overall method to perform the preprocessing steps for a given dataframe df.

        Args
            :df (pandas.DataFrame): a given DataFrame  which should be preprocessed.
            :rolling_window (str): the size of the rolling window (e.g. '1s', '1min', '1h').
            :sample_time (str): the size to use for resampling the DataFrame df. The resampling will be done on the date time column.

        Returns
            :df (pandas.DataFrame): the preprocessed DataFrame object.
        """

        try:
            # room n005 has NaN values for WIFI and BLE
            df[["WIFI", "BLE"]].fillna(value = 0, inplace = True)
        except:
            pass

        df = self.drop_na_rows(df)
        df = self.convert_features(df)

        if self.get_outliers_out:
            df = self.remove_outliers(df)

        df = self.add_weather_data(df)
        df = self.extract_room(df)
        df = self.remove_duplicates(df)
        df = self.remove_features(df)
        df = self.remove_invalid_values(df)
        df = self.create_rolling_windows(df, rolling_window = rolling_window, sample_time = sample_time)
        df = self.create_time_diff_features(df)
        df = self.create_new_features(df)
        df = self.fill_na(df)

        if self.date_time_column in df.columns:
            df = df.sort_values([self.date_time_column])
            # reset the index, since it is only int values. They should have a logical order.
            df = df.reset_index(drop = True)
        else:
            # if the date time column is the index
            df = df.sort_index()

        return df

    
    def drop_na_rows(self, df):
        """Remove rows where more than 90% of the columns in a data point are NA.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """
        try:
            threshold = 0.9
            # Remove rows with NA values exceeding the threshold of 90%
            df = df[df.isna().sum(axis=1) <= threshold]
        except Exception as e:
            print(e)
            pass
    
        return df
    

    def convert_features(self, df):
        """Convert the formats of features to their correct ones.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """
        
        if self.date_time_column in df.columns:
            try:
                df = df[df[self.date_time_column].notna()]
                df[self.date_time_column] = pd.to_datetime(df[self.date_time_column])
            except Exception as e:
                print(e)
                pass

        if "snr" in df.columns:
            try:
                df["snr"] = df["snr"].astype("float")
            except Exception as e:
                print(e)
                pass

        return df
    

    def remove_outliers(self, df, contamination:float = 0.075):
            """Entferne Ausreißer mit einem Isolation Forest.
        
        Args:
            df (pandas.DataFrame): DataFrame Objekt.
            contamination (float): relativer Anteil des Datensatzes, der als Ausreißer entfernt werden soll.

        Returns:
            df (pandas.DataFrame): DataFrame Objekt.
        """

            # n_estimators: wie viele Bäume sollen genutzt werden?
            # contamination: welcher relativer Anteil des Datensatzes soll als Ausreißer detektiert werden?
            # max_samples: mit wie vielen Datenpunkten soll der IsolationForest trainiert werden?
            isoforest = IsolationForest(n_estimators = 100, contamination = contamination, max_samples = int(df.shape[0]*0.8))
            # Isolation Forest auf den wichtigsten numerischen Werten durchführen (CO2, tmp, vis, hum und VOC).
            prediction = isoforest.fit_predict(df[["CO2", "tmp", "vis", "hum", "VOC"]])
            print("Number of outliers detected: {}".format(prediction[prediction < 0].sum()))
            print("Number of normal samples detected: {}".format(prediction[prediction >= 0].sum()))
            score = isoforest.decision_function(df[["CO2", "tmp", "vis", "hum", "VOC"]])
            df["anomaly_score"] = score
            # Zeilen mit anomaly_score < 0 werden vom Isolation Forest als Ausreißer interpretiert.
            df = df[df.anomaly_score >= 0]

            return df
    

    def add_weather_data(self, df):
        df_weather = pd.read_csv("wetterdaten_karlsruhe.csv")
        df_weather = df_weather.drop(columns = ["snow", "tsun"], axis = 1)

        df_weather = self.drop_na_rows(df_weather)
        df_weather = df_weather.fillna(0)

        df_weather["date"] = pd.to_datetime(df_weather["date"])
        df["date_time"] = pd.to_datetime(df["date_time"])

        df = pd.merge_asof(df.sort_values("date_time"), df_weather, left_on = "date_time", right_on = "date", direction = "nearest")

        df.drop(columns = ["date"], inplace = True)

        return df
    

    def extract_room(self, df):
        """Create new features for the Machine Learning training phase.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """

        try:

            df.loc[:, "room_number"] = df["device_id"].str.split("-").str[-1]
        
        except Exception as e:
            print(e)

        return df
    
    
    def remove_duplicates(self, df):
        """Remove duplicated rows from the data
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """
        # remove duplicated data points
        df = df.drop_duplicates([self.date_time_column, "room_number"])

        return df
    

    def remove_features(self, df):
        """Remove redundant features.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """

        columns = ["WIFI", "bandwidth", "channel_rssi", "channel_index", "device_id", "gateway", "f_cnt", "spreading_factor", 'rssi', "snr"]

        for col in columns:
            # iterate in case only some of those features are in the dataframe
            try:
                df.drop(columns = [col], axis = 1, inplace = True)
            except:
                continue
        
        return df
    

    def remove_invalid_values(self, df):
        """Remove invalid data points from the dataset.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """

        # humidity describes a percent value. Therefore, values over 100 are invalid
        df = df[df.hum <= 100]
        # temperature is measured in °C. Therefore, we assume there can't be room temperatures above 50°C.
        # this filter method might have to be reevaluated in case of fire outbreaks
        df = df[(df.tmp <= 50) & (df.tmp >= 10)]

        # remove data points with suspicious large value changes in VOC and CO2 over a short period of time (60 seconds)
        too_fast_VOC_rise = (df.VOC.diff() >= 1000) & (df[self.date_time_column].diff().dt.seconds < 60)
        too_fast_CO2_rise = (df.CO2.diff() >= 1000) & (df[self.date_time_column].diff().dt.seconds < 60)

        # Combine conditions
        suspicious_rises = too_fast_VOC_rise | too_fast_CO2_rise

        # Filter DataFrame
        df = df[~suspicious_rises]

        

        # The dataset contains data points with multiple zeroes. We assume they could result from false reading or resets of the sensor devices.
        df = df[(df.CO2 != 0) & (df.VOC != 0) & (df.tmp != 0) & (df.hum != 0)]
        # Thousands of data points include high inclines of the CO2 value while the other measured variables freeze (e.g. in some of them VOC stays at 450)
        invalid_values = (df.CO2 > 20000) & (df.VOC.diff() == 0) & (df.VOC.diff() == 0) & (df.BLE.diff() == 0) & (df.tmp.diff() == 0)
        df = df[invalid_values == False]

        df.reset_index(drop = True, inplace = True)

        return df
    
    
    def create_rolling_windows(self, df, rolling_window, sample_time = "1h"):
        """Resample the data on a given time interval (sample_time) and create rolling windows with the size of rolling_window.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.
            :rolling_window (str): the size of the rolling window (e.g. '1s', '1min', '1h').
            :sample_time (str): the size to use for resampling the DataFrame df. The resampling will be done on the date time column.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """
        df_sorted = df.sort_values(self.date_time_column)

        all_results = list()

        for room in df.room_number.unique():

            room_df = df_sorted[df_sorted.room_number == room]

            numerical_features = ["tmp","hum","CO2","VOC","vis","IR", "BLE", 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres']

            room_df = room_df.set_index(self.date_time_column)

            if self.roll:
                # resample the data points (sampling interval should be lower than the rolling window size)
                room_df_resampled = room_df[numerical_features].resample(sample_time).mean()
                # calculate the rolling windows
                room_df_rolled = room_df_resampled.rolling(rolling_window).mean()
                # reunite the non-numerical features with the data frame (date time values will not be considered)
                non_numerical_df = room_df.select_dtypes(exclude=['number']).resample(sample_time).first()
                result = room_df_rolled.join(non_numerical_df)
            else:
                result = room_df

            # remove NA values, calculate the time differences
            # result = self.preprocess_df(result.reset_index())

            result.loc[:, "room_number"] = room

            all_results.append(result)
            
        # concatenate all dataframes to one
        all_results_df = pd.concat([result_df for result_df in all_results if not result_df.empty])
        # remove empty rows for timestamps with no value
        all_results_df = self.drop_na_rows(all_results_df)

        return all_results_df
    

    def create_time_diff_features(self, df):
        """Create features which are related to time differences between data points.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """

        if self.date_time_column not in df.columns:
            # reset index as we need the column for the next taken methods (creating time difference features etc.)
            df = df.reset_index()

        df = df.sort_values(by = [self.date_time_column])
        df["time_diff_sec"] = df.groupby('room_number')[self.date_time_column].diff().dt.seconds

        # Iterate over each group
        # create new features which show the value changes compared to the previous data point
        numerical_features = ["tmp", "hum", "CO2", "VOC", "vis", "IR", "BLE", "vis", 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres']
        try:
            numerical_features.remove(self.label)
        except:
            pass

        for feature in numerical_features:
                df[f"{feature}_diff"] = df.groupby('room_number')[feature].diff()

        # fill 0 in case there is a division through zero
        df = df.replace([np.inf, -np.inf], 0)

        return df
    

    def fill_na(self, df):
        """Handle missing values in the data.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """

        for col in df.columns:
            is_na = df[col].isna().any()
            if is_na:
                if "_diff" in col:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(method = "bfill")
                    df[col] = df[col].fillna(method = "ffill")

        return df
    

    def create_new_features(self, df):
        """Add features related to date time and non-numerical ones like color.
        
        Args:
            :df (pandas.DataFrame): DataFrame object.

        Returns:
            :df (pandas.DataFrame): DataFrame object.
        """

        # we treat the year 2022 as 1, year 2023 as 2 etc.
        df.loc[:, "year"] = df[self.date_time_column].dt.year - 2021
        df.loc[:, "dayofweek"] = df[self.date_time_column].dt.dayofweek
        df.loc[:, "hour"] = df[self.date_time_column].dt.hour

        df.loc[:, "month"] = df[self.date_time_column].dt.month
        df.loc[df.month.isin([1,2,12]), "season"] = "winter"
        df.loc[df.month.isin([3,4,5]), "season"] = "spring"
        df.loc[df.month.isin([6,7,8]), "season"] = "summer"
        df.loc[df.month.isin([9,10,11]), "season"] = "fall"

        df.drop(columns = "month", axis = 1, inplace = True)

        # according to https://www.h-ka.de/fileadmin/Hochschule_Karlsruhe_HKA/Bilder_VW-EBI/HKA_VW-EBI_Anleitung_CO2-Ampeln.pdf
        # due to simplicity, we treat every value under 850 as green, as the CO2 value declines from 850 to 700 rapidly anyway.

        df.loc[(df.CO2 < 850), "color"] = "green"
        df.loc[(df.CO2 >= 850) & (df.CO2 < 1200), "color"] = "yellow"
        df.loc[(df.CO2 >= 1200) & (df.CO2 < 1600), "color"] = "red"
        df.loc[(df.CO2 >= 1600), "color"] = "red_blinking"

        # some rooms have different ratios between VOC and CO2.
        df["VOC_CO2_ratio"] = (df["VOC"]/df["CO2"]).round(4)
        # in case CO2 is 0
        df['VOC_CO2_ratio'] = df['VOC_CO2_ratio'].combine_first(df['VOC'])

        return df