import pandas as pd
import numpy as np

def clean_gross(x):
    try:
        if pd.isna(x): return np.nan
        return float(str(x).replace(",", "").strip())
    except:
        return np.nan

def recommend_genre(data_path="desafio_indicium_imdb.csv", top_n=5, verbose=True, out_csv="genre_recommendation.csv"):
    df = pd.read_csv(data_path)

    df['Gross_num'] = df['Gross'].apply(clean_gross)
    df['Runtime_min'] = df['Runtime'].str.extract(r'(\d+)').astype(float)
    df['Primary_Genre'] = df['Genre'].str.split(',').str[0].str.strip()

    genre_stats = df.groupby('Primary_Genre').agg(
        count=('Series_Title','count'),
        avg_gross=('Gross_num', lambda x: np.nanmean(x)),
        median_gross=('Gross_num', lambda x: np.nanmedian(x)),
        pct_with_gross=('Gross_num', lambda x: 100 * x.notna().mean()),
        avg_rating=('IMDB_Rating','mean'),
        avg_votes=('No_of_Votes','mean')
    ).reset_index()

    g = genre_stats.copy()
    g['avg_gross'] = g['avg_gross'].fillna(0)
    if g['avg_gross'].max() - g['avg_gross'].min() > 0:
        g['norm_avg_gross'] = (g['avg_gross'] - g['avg_gross'].min()) / (g['avg_gross'].max() - g['avg_gross'].min())
    else:
        g['norm_avg_gross'] = 0.0
    g['norm_avg_votes'] = (g['avg_votes'] - g['avg_votes'].min()) / (g['avg_votes'].max() - g['avg_votes'].min()) if g['avg_votes'].max() > g['avg_votes'].min() else 0.0
    g['norm_avg_rating'] = (g['avg_rating'] - g['avg_rating'].min()) / (g['avg_rating'].max() - g['avg_rating'].min()) if g['avg_rating'].max() > g['avg_rating'].min() else 0.0

    w_gross, w_votes, w_rating = 0.6, 0.25, 0.15
    g['composite_score'] = w_gross * g['norm_avg_gross'] + w_votes * g['norm_avg_votes'] + w_rating * g['norm_avg_rating']

    g_sorted = g.sort_values('composite_score', ascending=False).reset_index(drop=True)
    top_gross = df[['Series_Title','Genre','Gross_num','IMDB_Rating']].dropna(subset=['Gross_num']).sort_values('Gross_num', ascending=False).head(20)

    g_sorted.to_csv(out_csv, index=False)

    if verbose:
        print("Top genres by composite score:")
        print(g_sorted[['Primary_Genre','count','avg_gross','median_gross','pct_with_gross','avg_rating','avg_votes','composite_score']].head(top_n).to_string(index=False))
        print("\nTop grossing films:")
        print(top_gross.head(10).to_string(index=False))
        best = g_sorted.iloc[0]
        print("\nRecommendation:")
        print(f"-> Develop a film in PRIMARY GENRE: {best['Primary_Genre']}")
        print("Rationale: Highest composite score driven mostly by average gross. Consider big-budget action/adventure with franchise potential.")

    return g_sorted, top_gross

if __name__ == "__main__":
    recommend_genre()
