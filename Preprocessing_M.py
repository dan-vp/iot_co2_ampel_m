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