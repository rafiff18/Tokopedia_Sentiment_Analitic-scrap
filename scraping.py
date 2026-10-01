

import pandas as pd
import time
import random

try:
    from google_play_scraper import reviews, Sort
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    print("[ERROR] Jalankan: pip install google-play-scraper")

APP_ID   = 'com.tokopedia.tkpd'  #com.shopee.id kalo maw shopee 
APP_NAME = 'Tokopedia'
TARGET   = 10500                  
OUTPUT   = 'dataset_tokopedia.csv'

def scrape_reviews(app_id: str, app_name: str, target: int) -> pd.DataFrame:
    all_reviews = []
    continuation_token = None

    print(f"Mulai scraping '{app_name}' (target: {target:,} ulasan)...")
    print("-" * 50)

    while len(all_reviews) < target:
        try:
            batch, continuation_token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=200,
                continuation_token=continuation_token
            )

            for r in batch:
                all_reviews.append({
                    'review_id'  : r.get('reviewId', ''),
                    'username'   : r.get('userName', ''),
                    'review'     : r.get('content', ''),
                    'rating'     : r.get('score', 0),
                    'thumbs_up'  : r.get('thumbsUpCount', 0),
                    'date'       : str(r.get('at', ''))[:10],
                    'app_version': r.get('reviewCreatedVersion', ''),
                })

            print(f"  Terkumpul: {len(all_reviews):,} / {target:,}")

            if not continuation_token:
                print("  Tidak ada data lebih lanjut.")
                break

            time.sleep(random.uniform(0.5, 1.2))

        except Exception as e:
            print(f"  [ERROR] {e} — menunggu 10 detik...")
            time.sleep(10)

    df = pd.DataFrame(all_reviews[:target])
    return df


def label_sentiment(rating: int) -> str:
    if rating >= 4:
        return 'positif'
    elif rating == 3:
        return 'netral'
    else:
        return 'negatif'


def main():
    if not SCRAPER_AVAILABLE:
        print("Library tidak tersedia. Jalankan: pip install google-play-scraper")
        return

    # Scraping
    df = scrape_reviews(APP_ID, APP_NAME, TARGET)

    # Hapus review kosong & duplikat
    df = df[df['review'].str.strip().str.len() > 5]
    df = df.drop_duplicates(subset='review')
    df = df.dropna(subset=['review', 'rating'])
    df = df.reset_index(drop=True)

    # Labeling
    df['sentiment'] = df['rating'].apply(label_sentiment)

    # Ringkasan
    print("\n" + "=" * 50)
    print(f"HASIL SCRAPING - {APP_NAME}")
    print("=" * 50)
    print(f"Total ulasan : {len(df):,}")
    print(f"\nDistribusi Sentimen:")
    print(df['sentiment'].value_counts().to_string())
    print(f"\nDistribusi Rating:")
    print(df['rating'].value_counts().sort_index().to_string())
    print(f"\nRentang Tanggal: {df['date'].min()} s/d {df['date'].max()}")

    # Simpan
    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\nDataset tersimpan: {OUTPUT}")

    print("\nSample data:")
    print(df[['review', 'rating', 'sentiment']].head(5).to_string())

    return df


if __name__ == '__main__':
    main()
