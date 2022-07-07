import numpy as np
import pandas as pd

def clean_iris_data(df):
    df = df.drop(columns = ['species_id', 'measurement_id'])
    df = df.rename(columns = {'species_name': 'species'})
    dummy_df = pd.get_dummies(df[['species']], dummy_na= False, drop_first=True)
    df = pd.concat([df, dummy_df], axis = 1)
    
    return df

def split_iris_data(df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train_validate, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123, stratify=train_validate.survived)
    return train, validate, test

def prep_iris_data(df):
    df = clean_iris_data(df)
    train, validate, test = split_iris_data(df)
    return train, validate, test

# ----------------------------------------------------

def clean_titanic_data(df):
    df = df.drop_duplicates()
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    df = df.drop(columns = ['age', 'embarked', 'deck', 'class'])
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na = False, drop_first = [True,True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def split_titanic_data(df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train_validate, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123, stratify=train_validate.survived)
    return train, validate, test

def prep_titanic_data(df):
    '''
    The ultimate dishwasher
    '''
    df = clean_titanic_data(df)
    train, validate, test = split_titanic_data(df)
    return train, validate, test

def impute_mode(train, validate, test):
    '''
    Takes in train, validate, and test, and uses train to identify the best value to replace nulls in embark_town
    Imputes that value into all three sets and returns all three sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

# ----------------------------------------------------------------
def clean_telco_data(df):
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce") #conver the total charge into float
    df = df.drop_duplicates()
    df = df.dropna() # drop 11 row that total charge are null
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    
    # encode binary categorical variables into numeric values
    df['gender_encoded'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    df['online_security_bool'] = df.online_security.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df['online_backup_bool'] = df.online_backup.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    # Get dummies for non-binary categorical variables
    df_dummy = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, drop_first=True)
    df = pd.concat([df, df_dummy],axis=1)
    
    # encode number_relationships by utilizing information from dependents_encoded and partner_encoded
    df['number_relationships'] = df['dependents_encoded'] + df['partner_encoded']

    # encode number_online_services by utilizing information from online_security_encoded and online_backup_encoded
    df['number_online_services'] = df['online_security_bool'] + df['online_backup_bool']

    # encode tenure in years (rounded down) by utilizing information from tenure (currently stored in months)
    df['yearly_tenure'] = df.tenure.apply(lambda x: math.floor(x/12))

    # encode has_internet
    df['has_internet'] = df.internet_service_type.apply(lambda x: 0 if x == 'None' else 1)

    return df

def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.2, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def prep_telco_data(df):
    df = clean_telco_data(df)
    train, validate, test = split_telco_data(df)
    
    return train, validate, test
