import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# تنظیم Matplotlib برای پشتیبانی از فونت فارسی و RTL
plt.rcParams['font.family'] = 'Vazir'  # جایگزین با فونت فارسی موجود در سیستم
plt.rcParams['axes.unicode_minus'] = False

# خواندن و پیش‌پردازش دیتاست
try:
    data = pd.read_csv("imdb_top_1000.csv")
except FileNotFoundError:
    print(get_display(arabic_reshaper.reshape("خطا: فایل 'imdb_top_1000.csv' یافت نشد. لطفاً مطمئن شوید که فایل در مسیر درست قرار دارد.")))
    exit()

data['Genre'] = data['Genre'].str.split(', ')
data['IMDB_Rating'] = pd.to_numeric(data['IMDB_Rating'], errors='coerce')
data['Gross'] = pd.to_numeric(data['Gross'].str.replace(',', ''), errors='coerce')

# فیلتر کردن فیلم‌های با ژانر Biography
biography_movies = data[data['Genre'].apply(lambda x: 'Biography' in x if isinstance(x, list) else False)].copy()

if biography_movies.empty:
    print(get_display(arabic_reshaper.reshape("هیچ فیلمی با ژانر Biography یافت نشد.")))
    exit()

# ایجاد ماتریس ویژگی برای فیلم‌های Biography
biography_genres = set()
for movie_genres in biography_movies['Genre']:
    biography_genres.update(movie_genres)
biography_genres = list(biography_genres)

genre_matrix = np.zeros((len(biography_movies), len(biography_genres)))
for i, movie_genres in enumerate(biography_movies['Genre']):
    for genre in movie_genres:
        genre_matrix[i, biography_genres.index(genre)] = 1

# استانداردسازی امتیاز IMDB
scaler = StandardScaler()
scaled_ratings = scaler.fit_transform(biography_movies[['IMDB_Rating']].fillna(0))
features = np.hstack([genre_matrix, scaled_ratings])

# محاسبه ماتریس شباهت
similarity_matrix = cosine_similarity(features)

# تابع پیشنهاد فیلم
def recommend_movies(movie_title, n=5):
    try:
        idx = biography_movies[biography_movies['Series_Title'] == movie_title].index[0]
        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:n+1]
        
        title_text = arabic_reshaper.reshape(f"فیلم‌های پیشنهادی مشابه '{movie_title}':")
        print(get_display(title_text))
        print(get_display(arabic_reshaper.reshape("----------------------------------------")))
        for i, (movie_idx, score) in enumerate(scores):
            title = biography_movies.iloc[movie_idx]['Series_Title']
            year = biography_movies.iloc[movie_idx]['Released_Year']
            genres = ', '.join(biography_movies.iloc[movie_idx]['Genre'])
            rating = biography_movies.iloc[movie_idx]['IMDB_Rating']
            print(get_display(arabic_reshaper.reshape(f"{i+1}. {title} ({year})")))
            print(get_display(arabic_reshaper.reshape(f"   ژانر: {genres}")))
            print(get_display(arabic_reshaper.reshape(f"   امتیاز IMDB: {rating}")))
            print(get_display(arabic_reshaper.reshape(f"   شباهت: {score:.2f}")))
            print(get_display(arabic_reshaper.reshape("----------------------------------------")))
            
        plot_recommendations(movie_title, scores)
        
    except IndexError:
        print(get_display(arabic_reshaper.reshape(f"فیلم '{movie_title}' در پایگاه داده یافت نشد.")))
    except Exception as e:
        print(get_display(arabic_reshaper.reshape(f"خطا در اجرای تابع: {e}")))

# تابع تجسم
def plot_recommendations(movie_title, scores):
    titles = [biography_movies.iloc[idx]['Series_Title'] for idx, _ in scores]
    similarities = [sim for _, sim in scores]
    years = [biography_movies.iloc[idx]['Released_Year'] for idx, _ in scores]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(similarities)), similarities, marker='o', linestyle='-', color='blue')
    plt.xticks(range(len(titles)), [get_display(arabic_reshaper.reshape(title)) for title in titles], rotation=45, ha='right')
    plt.ylabel(get_display(arabic_reshaper.reshape('امتیاز شباهت')))
    plt.title(get_display(arabic_reshaper.reshape(f'فیلم‌های مشابه "{movie_title}"')))

    # افزودن برچسب سال انتشار بالای نقاط
    for i, (x, y) in enumerate(zip(range(len(similarities)), similarities)):
        plt.text(x, y + 0.01, str(years[i]), ha='center', fontsize=9)

    plt.tight_layout()
    plt.grid(True, linestyle='-', alpha=1)
    plt.show()

# تست تابع با یک فیلم نمونه
if __name__ == "__main__":
    recommend_movies("Schindler's List", n=5)


    
    