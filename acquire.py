import pandas as pd
import env

def new_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

import os

def get_titanic_data():
    filename = "titanic.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
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
        return pd.read_csv(filename)
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
        return pd.read_csv(filename)
    else:
        df = new_telco_data()
        df.to_csv(filename)
        return df