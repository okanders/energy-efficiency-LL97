import os
import pandas as pd
import numpy as np


def create_csv():
    # set the path to the folder containing the CSV files
    path_to_folder = '/Users/oliverkanders/Desktop/mlEcon/project1/proj1/code/ll84'

    # create an empty list to store the data frames
    cleaned_data = pd.DataFrame()

    # loop through the files in the folder
    for filename in os.listdir(path_to_folder):
        if filename.endswith('.csv'):
            # read in the CSV file
            file_path = os.path.join(path_to_folder, filename)
            df = pd.read_csv(file_path)

            print(filename)
            cleaned_df = clean_ll84(df)
            # add date:
            # extract the year from the file name
            year = filename.split("_")[-1].rstrip(".csv")
            year = int(year)
            # add a new 'year' column with the extracted year value
            cleaned_df['year'] = year

            # save the updated file back to disk with the year column included
           # updated_file_path = os.path.join(path_to_folder, file_name)
            #df.to_csv(updated_file_path, index=False)

            # append the data frame to the list
            cleaned_data = cleaned_data.append(cleaned_df)

    # concatenate the data frames in the list into a single data frame
    #combined_df = pd.concat(df_list, axis=0)

    # print the combined data frame
    cleaned_data.sort_values(['Property Id', 'year'], inplace=True)

    print(cleaned_data)

    # specify the file name for the CSV file
    file_name = 'll84.csv'

    # save the data frame as a CSV file in the same folder as the script
    cleaned_data.to_csv(file_name, index=False)

    print(f"The data frame has been saved as {file_name} in the same folder.")
    return cleaned_data



def commoncol():
    # read the two CSV files into data frames
    df1 = pd.read_csv('ll84_post_19_21.csv')
    df2 = pd.read_csv('pluto_post_19_21.csv')
    # get a list of column names for each data frame
    df1_cols = df1.columns.tolist()
    df2_cols = df2.columns.tolist()

    # check if there is any common column between the two data frames
    common_cols = [col for col in df1_cols if col in df2_cols]

    if common_cols:
        print('The two data frames share the following common column(s):', common_cols)
    else:
        print('The two data frames do not share any common columns.')

def bbl(df):
    # read the two CSV files into data frames
    #df1 = pd.read_csv('ll84_10_21.csv')
    #df2 = pd.read_csv('PLUTO_10_21.csv')
    # get a list of column names for each data frame
    #df1_cols = df1.columns.tolist()
    #df2_cols = df2.columns.tolist()
    #print(df1_cols[1])
    # assuming your DataFrame is named 'df'




    # remove hyphens and convert to integer
    df['BBL'] = df['NYC Borough, Block and Lot (BBL)'].apply(lambda s: ''.join(s.split('-')))
    df['BBL'] = pd.to_numeric(df['BBL'], errors='coerce')
    # replace 'Not Available' values with NaN
    df['BBL'] = df['BBL'].replace('Not Available', np.nan)
    # drop rows with missing values in 'BBL' column
    df.dropna(subset=['BBL'], inplace=True)
    # convert BBL column to integer
    df['BBL'] = df['BBL'].astype(int)
    # rename column to 'BBL'
    df.rename(columns={'BBL': 'BBL'}, inplace=True)
    df.to_csv('ll84_cl_pre_15_18.csv', index=False)









    #deal with Borough
    #df2['Borough'].replace('MN', 1, inplace=True)
    #df2['Borough'].replace('BX', 2, inplace=True)
    #df2['Borough'].replace('BK', 3, inplace=True)
    #df2['Borough'].replace('QN', 4, inplace=True)
    #df2['Borough'].replace('SI', 5, inplace=True)

    #Block
    # convert the column to a string and zero-pad the values for 5
    #df2['Block'] = df2['Block'].astype(str).str.zfill(5).astype(int)

    #Lot
    # convert the column to a string and zero-pad the values for 4

    #df2['Lot'] = df2['Lot'].astype(str).str.zfill(4).astype(int)

    #merge to make BBL
    #df2['BBL10'] = df2['Borough'].astype(str) + df2['Block'].astype(str).str.zfill(5) + df2['Lot'].astype(str).str.zfill(4)


def checkColumnType(df):
      # check the data type of a column
    column_type = df['BBL'].dtype

    if column_type == 'int64':
        print('The column contains integers.')
    elif column_type == 'float64':
        print('The column contains floats.')
    elif column_type == 'object':
        print('The column contains strings.')
    else:
        print('The column contains a different data type.')


def clean_ll84(df):
    if 'Property GFA - EPA Calculated (Buildings) (ft²)' in df.columns:
        # Rename the column
        df.rename(columns={'Property GFA - EPA Calculated (Buildings) (ft²)': 'Property GFA - Calculated (Buildings) (ft²)'}, inplace=True)

    # identify 6 features to keep
    features_to_keep = ['Property Id', 'Borough', 'Occupancy', 'Largest Property Use Type - Gross Floor Area (ft²)', 'Site EUI (kBtu/ft²)', 'ENERGY STAR Score', 'Electricity Use - Grid Purchase (kBtu)',
                    'Multifamily Housing - Gross Floor Area (ft²)', 
                    'Office - Gross Floor Area (ft²)', 
                    'Retail Store - Gross Floor Area (ft²)', 
                    'Non-Refrigerated Warehouse - Gross Floor Area (ft²)', 
                    'Manufacturing/Industrial Plant - Gross Floor Area (ft²)',
                    'Property GFA - Calculated (Buildings) (ft²)']

    # select the columns to keep using loc, and drop the rest
    df = df.loc[:, features_to_keep]

    # Drop observations with missing values in the 'Largest Property Use Type - Gross Floor Area (ft²)' column
    df = df.dropna(subset=['Largest Property Use Type - Gross Floor Area (ft²)', 'Property Id', 'Borough', 'Occupancy'])

    # replace all 'not available' with nan
    df = df.replace('Not Available', np.nan)

    # Convert columns to numeric data type and fill missing values with column means
    numeric_cols = ['Site EUI (kBtu/ft²)', 'ENERGY STAR Score', 'Electricity Use - Grid Purchase (kBtu)',
                    'Multifamily Housing - Gross Floor Area (ft²)', 
                    'Office - Gross Floor Area (ft²)', 
                    'Retail Store - Gross Floor Area (ft²)', 
                    'Non-Refrigerated Warehouse - Gross Floor Area (ft²)', 
                    'Manufacturing/Industrial Plant - Gross Floor Area (ft²)',
                    'Property GFA - Calculated (Buildings) (ft²)']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[numeric_cols] = df[numeric_cols].astype(int)

    #take log of EUI:
    #df['Source EUI (kBtu/ft²)'] = np.log(df['Source EUI (kBtu/ft²)'])
    df['Site EUI (kBtu/ft²)'] = np.log(df['Site EUI (kBtu/ft²)'])



        # Get the column names for the different types of spaces
    column_names = ['Multifamily Housing - Gross Floor Area (ft²)', 
                    'Office - Gross Floor Area (ft²)', 
                    'Retail Store - Gross Floor Area (ft²)', 
                    'Non-Refrigerated Warehouse - Gross Floor Area (ft²)', 
                    'Manufacturing/Industrial Plant - Gross Floor Area (ft²)']

    # Calculate the total floor area
    total_area = df['Property GFA - Calculated (Buildings) (ft²)']

    # Calculate the proportions for each type of space
    proportions = df[column_names].div(total_area, axis=0)

    # Add the proportions as a new column to the DataFrame
    # Add the proportions as new columns to the DataFrame
    df['Residential Proportion'] = proportions['Multifamily Housing - Gross Floor Area (ft²)']
    df['Office Proportion'] = proportions['Office - Gross Floor Area (ft²)']
    df['Retail Proportion'] = proportions['Retail Store - Gross Floor Area (ft²)']
    df['Storage Proportion'] = proportions['Non-Refrigerated Warehouse - Gross Floor Area (ft²)']
    df['Factory Proportion'] = proportions['Manufacturing/Industrial Plant - Gross Floor Area (ft²)']
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    return df




    #source_eui_mean = df['Source EUI (kBtu/ft²)'].mean()
    #site_eui_mean = df['Site EUI (kBtu/ft²)'].mean()
    #energystar_mean = df['Site EUI (kBtu/ft²)'].mean()
    #electric_mean = df['Electricity Use - Grid Purchase (kBtu)'].mean()
    # Fill in the means for missing values in the 'column1' and 'column2' columns
    #df['Source EUI (kBtu/ft²)'] = df['Source EUI (kBtu/ft²)'].fillna(source_eui_mean)
    #df['Source EUI (kBtu/ft²)'] = df['Source EUI (kBtu/ft²)'].fillna(site_eui_mean)
    #df['Source EUI (kBtu/ft²)'] = df['Source EUI (kBtu/ft²)'].fillna(energystar_mean)
    #df['Source EUI (kBtu/ft²)'] = df['Source EUI (kBtu/ft²)'].fillna(electric_mean)



    # fill NaN values with column means


def merge(df1,df2):
    # read the two CSV files into data frames

    # merge the two data frames on a common column
    merged_df = pd.merge(df1, df2, on='BBL10', how='left')

    # write the merged data frame to a new CSV file
    merged_df.to_csv('post_file.csv', index=False)


def split_dataset(panel_data):
    # Load the panel dataset

    # Convert the year column to a datetime object
    panel_data['year'] = pd.to_datetime(panel_data['year'], format='%Y')

    # Subset the data frame based on the desired time periods
    buildings_2015_2019 = panel_data[(panel_data['year'] >= '2015-01-01') & (panel_data['year'] <= '2019-12-31')]

    buildings_2020_2021 = panel_data[(panel_data['year'] >= '2020-01-01') & (panel_data['year'] <= '2022-12-31')]

    # Save the data frames as separate CSV files
    buildings_2015_2019.to_csv('buildings_2015_2019.csv', index=False)

    buildings_2020_2021.to_csv('buildings_2020_2021.csv', index=False)



def main():
   #create csv
    my_data = create_csv()
    #split dataset according to type
    split_dataset(my_data)

    

  

# call the main function
if __name__ == '__main__':
    main()
