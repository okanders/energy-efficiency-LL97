import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV  # split dataset into train and test for cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import tensorflow as tf
from tensorflow import keras
import seaborn as sns



def linearTimeTrend(df):
    # extract the year component and convert to integer
    df['year'] = pd.to_datetime(df['year'])
    df['year'] = df['year'].dt.year.astype(int)

    # create a linear time trend variable
    df['year'] = df['year'] - 2015
    return df

def fixed_effects(cleaned_data, features):
    # Create dummy variables for the 'Borough' feature
    borough_dummies = pd.get_dummies(cleaned_data['Borough'], prefix='Borough')

    # Create dummy variables for the proportion columns in bins
    proportions = ['Residential Proportion', 'Office Proportion', 'Retail Proportion', 'Storage Proportion', 'Factory Proportion']
    proportion_dummies = pd.DataFrame()
    for proportion in proportions:
        bins = pd.cut(cleaned_data[proportion], bins=[-0.01, 0, 0.25, 0.5, 0.75, 1], labels=['None', 'Low', 'Medium', 'High', 'Full'])
        proportion_dummies_temp = pd.get_dummies(bins, prefix=proportion)
        proportion_dummies = pd.concat([proportion_dummies, proportion_dummies_temp], axis=1)


    dummies = list(borough_dummies.columns) + list(proportion_dummies.columns)
    # Add the dummy variables to the DataFrame
    cleaned_data = pd.concat([cleaned_data[features], borough_dummies, proportion_dummies], axis=1)

    # Remove the original 'Borough', proportion, and year columns from the DataFrame
    cleaned_data.drop(['Borough'] + proportions , axis=1, inplace=True)

    return cleaned_data, dummies

def ols(X,y):

    # add intercept term to X
    X = sm.add_constant(X)

    # fit OLS regression model
    model = sm.OLS(y, X).fit()

    print(model.summary())

    # extract coefficients and p-values for each predictor
    coef_df = pd.DataFrame({'Predictor': X.columns, 'Coefficient': model.params, 'P-value': model.pvalues})

    # separate out the dummy variables from the continuous predictors
    continuous_predictors = ['const'] + list(X.select_dtypes(include=['float64', 'int64']).columns)
    dummy_predictors = list(set(X.columns) - set(continuous_predictors))

    # print out the coefficients and p-values for the continuous predictors
    print('\nContinuous Predictors:\n')
    print(coef_df[coef_df['Predictor'].isin(continuous_predictors)].to_string(index=False))

    # print out the coefficients and p-values for the dummy predictors
    print('\nDummy Predictors:\n')
    print(coef_df[coef_df['Predictor'].isin(dummy_predictors)].to_string(index=False))

    return model




    #from linearmodels.panel import PanelOLS

    # set up the model
    #exog_vars = ['const', 'Occupancy', 'Property Type - Gross Floor Area (ft²)', 'ENERGY STAR Score', 'Electricity Use - Grid Purchase (kBtu)']
    #exog = sm.add_constant(df[exog_vars + ['year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019', 'year_2020']])
    #mod = PanelOLS(df['y'], exog, entity_effects=True, time_effects=True)

    # estimate the model
    #res = mod.fit(cov_type='clustered', cluster_entity=True)

def run_ols(df):
         # Remove NaNs
    df = df.dropna()
    
    # Replace inf values with NaNs and remove them
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Make a copy of the original data
    features_to_keep = ['Property Id', 'Borough', 'Occupancy', 'Largest Property Use Type - Gross Floor Area (ft²)', 
                    'Site EUI (kBtu/ft²)', 'ENERGY STAR Score', 'Electricity Use - Grid Purchase (kBtu)',
                    'Multifamily Housing - Gross Floor Area (ft²)', 
                    'Office - Gross Floor Area (ft²)', 
                    'Retail Store - Gross Floor Area (ft²)', 
                    'Non-Refrigerated Warehouse - Gross Floor Area (ft²)', 
                    'Manufacturing/Industrial Plant - Gross Floor Area (ft²)',
                    'Property GFA - Calculated (Buildings) (ft²)',
                    'Residential Proportion',
                    'Office Proportion',
                    'Retail Proportion',
                    'Storage Proportion',
                    'Factory Proportion',
                    'year']

    df, dummies = fixed_effects(df, features_to_keep)

    df = linearTimeTrend(df)


    predictors = ['Property Id', 'Occupancy', 'Largest Property Use Type - Gross Floor Area (ft²)',
                  'ENERGY STAR Score', 'Electricity Use - Grid Purchase (kBtu)', 'year'] + dummies
    target = 'Site EUI (kBtu/ft²)'
    # show summary statistics
    #print(df.describe())



   # correlation_matrix = df.corr()
    #sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')

    #Show the plot
    #plt.show()

    # display the correlation matrix    
    #print(correlation_matrix)


    # plot histograms of each column
    #df.hist()
    #plt.show()    

    X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=42)


      # Run OLS regression on training data
    model = ols(X_train, y_train)

    # Create scatter plot of predicted vs actual values on test data
    y_pred = model.predict(sm.add_constant(X_test))
    y_pred = remove_outliers(y_pred, y_test)


    plt.scatter(y_pred, y_test)
    plt.xlabel('Predicted Values of EUI (kBtu/Sq^2')
    plt.ylabel('Actual Values of EUI (kBtu/Sq^2)')
    plt.title('Post, 2020-2021: OLS Regression Results')
    plt.show()


def remove_outliers(y_pred, y_test):
    # Calculate IQR of test set
    Q1 = np.percentile(y_test, 25)
    Q3 = np.percentile(y_test, 75)
    IQR = Q3 - Q1

    # Calculate lower and upper bounds
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # Find indices of outliers in y_pred
    outlier_indices = np.where((y_pred < lower_bound) | (y_pred > upper_bound))

    # Replace outliers with mean of y_test
    y_pred[outlier_indices] = np.mean(y_test)

    return y_pred


def olsD(y, constant):
   # create a dataframe with the random constant x
    print('y ', y.shape)
    print('con', constant)
    if isinstance(y, np.ndarray):
        x = pd.DataFrame({'x': constant}, index=range(len(y)))
    else:
        x = pd.DataFrame({'x': constant}, index=y.index)


    #add intercept term to x
    x = sm.add_constant(x)

    #fit OLS regression model
    model = sm.OLS(y, x).fit()
    coefficients = model.params

    return coefficients

def NN(df):
     # Remove NaNs
    df = df.dropna()
    
    # Replace inf values with NaNs and remove them
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Make a copy of the original data
    features_to_keep = ['Property Id', 'Borough', 'Occupancy', 'Largest Property Use Type - Gross Floor Area (ft²)', 
                    'Site EUI (kBtu/ft²)', 'ENERGY STAR Score', 'Electricity Use - Grid Purchase (kBtu)',
                    'Multifamily Housing - Gross Floor Area (ft²)', 
                    'Office - Gross Floor Area (ft²)', 
                    'Retail Store - Gross Floor Area (ft²)', 
                    'Non-Refrigerated Warehouse - Gross Floor Area (ft²)', 
                    'Manufacturing/Industrial Plant - Gross Floor Area (ft²)',
                    'Property GFA - Calculated (Buildings) (ft²)',
                    'Residential Proportion',
                    'Office Proportion',
                    'Retail Proportion',
                    'Storage Proportion',
                    'Factory Proportion',
                    'year']

    df, dummies = fixed_effects(df, features_to_keep)

    df = linearTimeTrend(df)


    predictors = ['Property Id', 'Occupancy', 'Largest Property Use Type - Gross Floor Area (ft²)',
                  'ENERGY STAR Score', 'Electricity Use - Grid Purchase (kBtu)', 'year'] + dummies
    target = 'Site EUI (kBtu/ft²)'
    # show summary statistics
    #print(df.describe())



   # correlation_matrix = df.corr()
    #sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')

    #Show the plot
    #plt.show()

    # display the correlation matrix    
    #print(correlation_matrix)


    # plot histograms of each column
    #df.hist()
    #plt.show()    

    X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=42)




    #ols(X_train, y_train)





    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    # Define the architecture of the neural network
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model on the training data
    history = model.fit(X_train_scaled, y_train, epochs=2, validation_split=0.2)

    # Evaluate the model on the test data
    test_loss = model.evaluate(X_test_scaled, y_test)
    print(f'Test loss: {test_loss:.4f}')



    # Plot the distribution of the target variable
    #plt.hist(y_test, bins=50)
    #plt.xlabel('Site EUI (kBtu/ft²)')
    #plt.ylabel('Frequency')
    #plt.title('Distribution of Site EUI (kBtu/ft²) in Test Set')
    #plt.show()

    # Predict the Site EUI values for the test set
    y_pred = model.predict(X_test_scaled)


    y_pred = remove_outliers(y_pred, y_test)


    #Plot the predicted versus actual values
    #plt.scatter(y_test, np.exp(y_pred))
    #plt.xlabel('Actual Site EUI (kBtu/ft²)')
    #plt.ylabel('Predicted Site EUI (kBtu/ft²)')
    #plt.title('Predicted vs Actual Site EUI (kBtu/ft²)')
    #plt.show()

    return y_test, y_pred





def main():

    #Read the data into a DataFrame
    pre_data = pd.read_csv('buildings_2015_2019.csv')
    post_data = pd.read_csv('buildings_2020_2021.csv')

    #run_ols(pre_data)
    #run_ols(post_data)

    pre_test, pre_pred = NN(pre_data)
    print('test, pred', pre_test.shape, pre_pred.shape)
    post_test, post_pred = NN(post_data)

    #set the seed for reproducibility
    np.random.seed(40)
    
    #generate a random constant
    constant = np.random.randn()

    pre_test_slope = olsD(pre_test, constant)
    pre_pred_slope = olsD(pre_pred, constant)
    post_test_slope = olsD(post_test, constant)
    post_pred_slope = olsD(post_pred, constant)

    # Create a scatter plot of the pre and post data
    plt.scatter(pre_test, pre_pred, label='Pre Data')
    plt.scatter(post_test, post_pred, label='Post Data')

    # Add the OLS line
    #x = np.array([16])
    x = np.arange(16)
    pre_slope = pre_test_slope[0] - pre_pred_slope[0]
    post_slope = post_test_slope[0] - post_pred_slope[0]
    DiD = post_slope-pre_slope
    print('DiD', DiD)

    print(x.shape)
    print(constant)
    print(pre_slope.shape)
    print(post_slope.shape)
    #print('hmm', np.dot(pre_slope, x))
    plt.plot(x, constant +  DiD * x, color='black', label='OLS')

    plt.xlabel('Actual Site EUI (kBtu/ft²)')
    plt.ylabel('Predicted Site EUI (kBtu/ft²)')
    plt.title('DiD: Demonstrating Differences Across Periods')
    plt.legend()
    plt.show()


# call the main function
if __name__ == '__main__':
    main()
