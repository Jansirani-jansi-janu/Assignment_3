import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()

    df['sex'] = le_sex.fit_transform(df['sex'])
    df['smoker'] = le_smoker.fit_transform(df['smoker'])
    df['region'] = le_region.fit_transform(df['region'])

    return df
