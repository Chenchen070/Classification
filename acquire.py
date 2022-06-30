import pandas as pd
import env
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

# -----------------------------acquire data------------------------------------------
def new_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

import os

def get_titanic_data():
    filename = "titanic.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    else:
        df = new_titanic_data()
        df.to_csv(filename)
        return df 


def new_iris_data():
    return pd.read_sql('select * from measurements left join species using (species_id)', 
                       get_connection('iris_db'))

def get_iris_data():
    filename = "iris.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    else:
        df = new_iris_data()
        df.to_csv(filename)
        return df 

def new_telco_data():
    return pd.read_sql('select * from customers left join internet_service_types using(internet_service_type_id) left join payment_types using (payment_type_id) left join contract_types using (contract_type_id)',
                      get_connection('telco_churn'))

def get_telco_data():
    filename = "telco.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    else:
        df = new_telco_data()
        df.to_csv(filename)
        return df

# ---------------------------------prep data----------------------------------

def prep_iris(df):
    df = df.drop(columns = ['species_id', 'measurement_id'])
    df = df.rename(columns = {'species_name': 'species'})
    dummy_df = pd.get_dummies(df[['species']], dummy_na= False, drop_first=True)
    df = pd.concat([df, dummy_df], axis = 1)
    
    return df

def prep_titanic(df):
    df = df.drop_duplicates()
    df = df.drop(columns = ['age', 'embarked', 'deck', 'class'])
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na = False, drop_first = [True,True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco(df):
    df = df.drop_duplicates()
    cols_to_drop = ['contract_type_id', 'payment_type_id', 'internet_service_type_id', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv','streaming_movies', 'paperless_billing']
    df = df.drop(columns = cols_to_drop)
    df_dummy = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'churn']], dummy_na=False, drop_first=[True,True,True,True,True,True])
    df = pd.concat([df, df_dummy],axis=1)
    return df

