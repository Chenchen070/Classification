import pandas as pd
import env

def get_connection(db, username=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

def get_iris_data():
    return pd.read_sql('select * from measurements left join species using (species_id)', 
                       get_connection('iris_db'))

def get_telco_data():
    return pd.read_sql('select * from customers left join internet_service_types using(internet_service_type_id) left join payment_types using (payment_type_id) left join contract_types using (contract_type_id)',
                      get_connection('telco_churn'))

