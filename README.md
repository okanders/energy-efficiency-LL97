Introduction
Environmental sustainability has arrived at the forefront of policy in New York City over the past decade as water levels and temperatures climb. 
Responsible for around 36 percent of the world’s energy use, commercial and residential buildings contribute immensely to this struggle. In 2019, Local Law 
97, one of New York City’s more ambitious plans against climate change was passed: incentivizing buildings over 25,000 square feet to meet new energy 
efficiency standards at the cost of higher taxes. Investigating this policy, I will hope to test its true effectiveness three years into its implementation 
(with a milestone of 40 percent reduction in greenhouse gas emissions by 2030).  By employing data on housing characteristics, location, temperature, and 
energy output, I will generate many different estimators (OLS, Fixed Effects, Ridge Regression, and Neural Networks) for energy use in building complexes 
preceding Local Law 97. I can then use my model to predict energy consumption after Local Law 97 against realized outputs to reveal the effectiveness of 
the policy. Inspired by work from Constantine Kontokosta’s A Data-Driven Predictive Model of City-Scale Energy Use in Buildings, I hope to exercise many 
machine learning methods to explain with greater insight the relationships between housing features and energy output in New York City, particularly if any 
movement has been spurred by local proactivity in Local Law 97. 

Similar to Kontokosta, I selected optimal features of Borough, Occupancy, and Proportions of Residential, Office, Storage, and Factory square footage to 
the gross square footage of the building. Kontokosta engaged in a more rigorous effort to determine these powerful features through a fitting series of OLS 
models. Employing his selected features, I differed in choosing a Neural Network model of dense layers to describe the relationship of the features to Site 
Energy Use Intensity (kBtu/ft2), my target variable. I ultimately took the natural log of the EUI, assuming a normal relationship between my variables for 
ease of description. 

Prompt:
Particularly focusing on residential housing in New York City, employing housing characteristics, location, and temperature, we will investigate the 
different predictors of energy output before the implementation of Local Law 97: ultimately, comparing our predictor to actual outputs in the years 
following the passing of Local Law 97.
Data Sources and Descriptions
I used the Energy and Water Disclosure for Local Law 84 (EWD: coinciding bill to Local Law 97 for disclosure of outputs and housing characteristics for 
complexes over 25,000 square feet). 
Note: After discussing the project in greater depth with Giulia, I decided to not merge an alternate PLUTO dataset on housing features, and would instead 
pursue solely the EWD. 



In processing, I focused on feature selection, and development of key measures: proportional space compared to gross.

I would regress the data with a Neural Network of 6 Dense layers, with an Adams optimizer, MSE loss function and learning rate of .001. 
I would also created a threshold as to determine outliers in predicted and testing data before plotting.
The OLS regressions failed to capture the complexity of the dataset, which may have been a limiting factor in deriving results because of the high dimensionality. Clustering would have been a useful technqiue and should be applied in future.
