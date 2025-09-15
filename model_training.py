"""model_training.py

Treina um modelo para prever IMDB_Rating a partir do dataset fornecido.
Uso: python model_training.py --data path/to/desafio_indicium_imdb.csv --output imdb_rating_model.pkl
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
import re

def clean_gross(x):
    try:
        if pd.isna(x): return np.nan
        return float(str(x).replace(',', '').strip())
    except:
        return np.nan

def preprocess_df(df):
    df = df.copy()
    df['Gross_num'] = df['Gross'].apply(clean_gross)
    df['Runtime_min'] = df['Runtime'].str.extract(r'(\d+)').astype(float)
    years = df['Released_Year'].astype(str).str.extract(r'(\d{4})')
    df['Year_num'] = pd.to_numeric(years[0], errors='coerce')
    df['Meta_score_filled'] = df['Meta_score'].fillna(df['Meta_score'].median())
    df['Primary_Genre'] = df['Genre'].astype(str).str.split(',').str[0].str.strip()
    return df

def top_director_transform(X, top_n=20):
    if hasattr(X, 'iloc'):
        s = X.iloc[:,0]
    else:
        import pandas as pd
        s = pd.Series([v[0] if isinstance(v, (list, tuple, np.ndarray)) else v for v in X.ravel()])
    top = s.value_counts().head(top_n).index.tolist()
    return s.apply(lambda val: val if val in top else 'Other').values.reshape(-1,1)

def build_pipeline(X_train):
    numeric_features = ['Runtime_min','Meta_score_filled','No_of_Votes','Gross_num','Year_num']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Primary_Genre','Certificate']
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    top_dirs = X_train['Director'].value_counts().head(20).index.tolist()
    def reduce_director(arr):
        import pandas as pd, numpy as np
        if isinstance(arr, pd.DataFrame):
            s = arr.iloc[:,0]
        else:
            s = pd.Series(arr.ravel())
        return s.apply(lambda x: x if x in top_dirs else 'Other').values.reshape(-1,1)

    director_transformer = Pipeline([
        ('reduce', FunctionTransformer(reduce_director, validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    text_transformer = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=200, stop_words='english'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('dir', director_transformer, ['Director']),
        ('txt', text_transformer, 'Overview')
    ], remainder='drop')

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    return model

def train_and_save(data_path, output_path):
    df = pd.read_csv(data_path)
    df = preprocess_df(df)
    features = ['Runtime_min','Meta_score_filled','No_of_Votes','Gross_num','Year_num','Primary_Genre','Certificate','Overview','Director']

    df_model = df[features + ['IMDB_Rating']].copy().dropna(subset=['IMDB_Rating'])
    X = df_model[features]
    y = df_model['IMDB_Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, output_path)
    metrics = {'rmse': float(rmse), 'r2': float(r2), 'model': 'RandomForestRegressor'}
    with open('summary.json', 'w') as f:
        import json
        f.write(json.dumps(metrics, indent=4))

    print(f"Model saved to {output_path}")
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train model for IMDB rating prediction')
    parser.add_argument('--data', type=str, default='desafio_indicium_imdb.csv', help='path to csv data')
    parser.add_argument('--output', type=str, default='imdb_rating_model.pkl', help='output model file path')
    args = parser.parse_args()
    train_and_save(args.data, args.output)

if __name__ == '__main__':
    main()
