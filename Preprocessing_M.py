import os, io
import pandas as pd
import tarfile, zipfile
import numpy as np
import time


class Data_Extractor:

    def __init__(self, first_directory, directory):
        self.first_directory = first_directory
        self.directory = directory


    def create_df(self):
        self.extract_zip_file(self.first_directory)
        self.extract_zip_files(self.directory)
        self.delete_zip_files(self.directory)
        self.df = self.get_data(self.directory)

        return self.df
    
    def extract_zip_file(self, directory):
        try:
            with zipfile.ZipFile(directory, 'r') as zip_ref:
                zip_ref.extractall(directory)
        except Exception as e:
                print(e)


    def extract_zip_files(self, directory):
        """Extract all zip files in the parameter directory."""

        for file_name in os.listdir(directory):
            if file_name.endswith(".zip"):
                try:
                    path = os.path.join(directory, file_name)
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(directory)
                        print(f"Extracted {file_name} in {directory}")
                except Exception as e:
                    print(e)


    def delete_zip_files(self, directory):
        """Delete all zip files in a directory"""
        try:
            for file_name in os.listdir(directory):
                if file_name.endswith(".zip"):
                    path = os.path.join(directory, file_name)
                    os.remove(path)
                    print(f"File {file_name} has been deleted.")
        except Exception as e:
            print(e)


    def get_tar_files(self, path, df):
        """A recursive function to extract the files from the tar folders. Each file contains the measurement 
        of the so called 'CO2-Ampeln' (CO2 traffic lights) at a specific room at a specific day."""
        folder_names = list()

        # tar folders can not be accessed like usual folders. Instead, it is recommended to use a package like 'tarfile' to extract their information
        if path.split(".")[-1] == "tar":
            try:
                # rename the folder in case it is a normal one but contains '.tar' anyway
                os.listdir(path)
                os.rename(path, path.replace(".tar", ""))
                path = path.replace(".tar", "")
            except:
                i = 1
                with tarfile.open(path, "r") as tar:
                    df_list = []
                    for file in tar.getmembers():
                        # extract the DAT file
                        dat_data = tar.extractfile(file)
                
                        # Read the DAT file into a pandas DataFrame
                        # header = 1 to avoid a false format of the DataFrame
                        # ";" is the separator letter
                        try:
                            df_new = pd.read_csv(io.BytesIO(dat_data.read()), delimiter=';', 
                                                header = 1, encoding='unicode_escape', on_bad_lines='skip')
                        except Exception as e:
                            print(e)
                            print("Error for file " + str(file))
                            pass
                        # store the information of each file in the dataframe 'df'
                        df_list.append(df_new)
                        
                try:
                    # merge the dataframes
                    df_all_new = pd.concat(df_list, ignore_index = True)
                    df = pd.concat([df, df_all_new], ignore_index = True)
                except Exception as e:
                    print(e)
        else:
            names = os.listdir(path)

            for name in names:
                # check for folder and file names
                if len(name.split(".")[-1]) > 1:
                    folder_names.append(name)

            for folder in folder_names:
                df = self.get_tar_files(path + f"/{folder}", df)

        return df


    def get_data(self, directory):
        """Open the tar files and put everything into a single dataframe."""
        df = pd.DataFrame()

        t = time.time()
        building_names = list()

        for name in os.listdir(directory):
            building_names.append(name.split("-")[-1])
            print(name)

            df = self.get_tar_files(directory + f"/{name}", df)
            print(round(time.time() - t, 2))

        return df
    




    class Data_Preprocessing:

        def __init__(self, get_outliers_out = True):
            self.get_outliers_out = get_outliers_out


        def preprocess_df(self, df):
            """Overall method to perform the preprocessing steps for a given dataframe df."""
            
            df = self.convert_features(df)
            df = self.create_new_features(df)
            df = self.remove_duplicates(df)
            df = self.remove_features(df)
            df = self.remove_invalid_values(df)
            if self.get_outliers_out:
                df = self.remove_outliers(df)
            df = self.create_average_differentials(df)

            df = self.fill_na(df)

            df = df.sort_values(["date_time"])
            # reset the index, since it is only int values. They should have a logical order.
            df = df.reset_index(drop = True)

            return df
        
        
        def remove_duplicates(self, df):
            # remove duplicated data points
            df = df.drop_duplicates(["date_time", "room_number"])
            return df
        

        def fill_na(self, df):
            """Handle missing values in the data."""

            for col in df.columns:
                is_na = df[col].isna().any()
                if is_na:
                    if "_diff" in col:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna("bfill")
                        df[col] = df[col].fillna("ffill")

            return df


        def convert_features(self, df):
            """Convert the formats of features to their correct ones."""

            df["date_time"] = pd.to_datetime(df["date_time"])
            df["snr"] = df["snr"].astype("float")
            return df


        def remove_features(self, df):
            """Remove redundant features."""

            df.drop(columns = ["WIFI", "bandwidth", "channel_rssi", "channel_index", "device_id", "gateway", "f_cnt", "spreading_factor"], axis = 1, inplace = True)
            
            return df


        def remove_invalid_values(self, df):
            """Remove invalid data points from the dataset."""

            # humidity describes a percent value. Therefore, values over 100 are invalid
            df = df[df.hum <= 100]
            # temperature is measured in °C. Therefore, we assume there can't be room temperatures above 50°C.
            # this filter method might have to be reevaluated in case of fire outbreaks
            df = df[df.tmp <= 50]

            # The dataset contains data points with multiple zeroes. We assume they could result from false reading or resets of the sensor devices.
            df = df[(df.CO2 != 0) & (df.VOC != 0) & (df.tmp != 0) & (df.hum != 0)]
            # Thousands of data points include high inclines of the CO2 value while the other measured variables freeze (e.g. in some of them VOC stays at 450)
            invalid_values = (df.CO2 > 20000) & (df.VOC_diff == 0) & (df.VOC_diff == 0) & (df.BLE_diff == 0) & (df.tmp_diff == 0)
            df = df[invalid_values == False]

            df.reset_index(drop = True, inplace = True)

            return df


        def create_new_features(self, df):
            """Create new features for the Machine Learning training phase."""

            try:
                # according to https://www.h-ka.de/fileadmin/Hochschule_Karlsruhe_HKA/Bilder_VW-EBI/HKA_VW-EBI_Anleitung_CO2-Ampeln.pdf
                # due to simplicity, we treat every value under 850 as green, as the CO2 value declines from 850 to 700 rapidly anyway.
                df.loc[(df.CO2 < 850), "color"] = "green"
                df.loc[(df.CO2 >= 850) & (df.CO2 < 1200), "color"] = "yellow"
                df.loc[(df.CO2 >= 1200) & (df.CO2 < 1600), "color"] = "red"
                df.loc[(df.CO2 >= 1600), "color"] = "red_blinking"

                # we treat the year 2022 as 1, year 2023 as 2 etc.
                df["year"] = df.date_time.dt.year - 2021

                df.loc[:, "room_number"] = df["device_id"].str.split("-").str[-1]
                df.loc[:, "building_name"] = df["room_number"].str.replace('\d+', '', regex = True)

                building_dict = {"ama":"am","amb":"am", 
                                    "ba":"b", "bb":"b",
                                    "eu":"e",
                                    "fa":"f","fu":"f",
                                    "lia":"li","lib":"li","lie":"li","liu":"li",
                                    "mu":"m"}
                
                df.replace({"building_name": building_dict}, inplace = True)
            
            except Exception as e:
                print(e)
                pass

            df.sort_values(by = ["date_time"]).reset_index(drop = True, inplace = True)

            df["time_diff_sec"] = df.groupby('room_number')["date_time"].diff().dt.seconds

            # Iterate over each group
            # create new features which show the value changes compared to the previous data point
            for feature in ["tmp", "hum", "CO2", "VOC", "vis", "IR", "BLE", "vis"]:
                    df[f"{feature}_diff"] = df.groupby('room_number')[feature].diff()

            df.fillna(0, inplace = True)

            df["dayofweek"] = df["date_time"].dt.dayofweek
            df["hour"] = df["date_time"].dt.hour
            df["month"] = df["date_time"].dt.month

            df.drop(columns = ["building_name"], axis = 1, inplace = True)


            return df


        def remove_outliers(self, df):
                """Remove outliers for CO2, VOC, tmp and hum."""

                for feature in ["CO2_diff", "VOC_diff", "tmp_diff", "hum_diff"]:
                        threshold = df[feature].mean() + 4*df[feature].std()
                        df = df[df[feature].abs() <= threshold]

                df.reset_index(drop = True, inplace = True)

                # since outliers (data points) were removed, the measured differences of variable changes between time points of a specific room are not accurate. They have to be calculated again.
                # The differences were required to estimate invalid values or outliers. Thus it has been implemented this way.
                df = self.create_new_features(df)

                return df

        
        def create_average_differentials(self, df):
            """Calculate the average differentials per second for CO2, VOC, tmp, hum, IR and vis."""

            for feature in ["CO2_diff", "VOC_diff", "tmp_diff", "hum_diff", "IR_diff", "vis_diff"]:
                    df[f"{feature}_per_sec"] = df[feature].div(df["time_diff_sec"])

            return df