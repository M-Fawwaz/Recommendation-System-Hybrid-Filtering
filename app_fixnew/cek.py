import pandas as pd
import numpy as np

# Membaca data
ratings = pd.read_csv('data/dataset/ratings_restaurant.csv')

# Mengambil seluruh Nama Restoran yang ada di dataset
all_restaurants = ratings['Nama Restoran'].unique()

# Menghitung total item (jumlah Nama Restoran unik)
total_restaurants = len(all_restaurants)

print("Total Nama Restoran unik di dataset:", total_restaurants)

# Menghitung rata-rata rating per restoran
average_ratings = ratings.groupby('Nama Restoran')['rating'].mean().reset_index()
average_ratings.columns = ['Nama Restoran', 'Average Rating']

# Membulatkan dan mengatur rentang rating
average_ratings['Average Rating'] = average_ratings['Average Rating'].round().clip(1, 5)

# Mengurutkan restoran berdasarkan rating rata-rata
average_ratings = average_ratings.sort_values(by='Average Rating', ascending=False)

# Menampilkan restoran dengan rating rata-rata tertinggi dan terendah
top_restaurants = average_ratings.head(10)  # Menampilkan 10 restoran teratas
bottom_restaurants = average_ratings.tail(10)  # Menampilkan 10 restoran terbawah

print("Restoran dengan rating rata-rata tertinggi:")
print(top_restaurants)

print("Restoran dengan rating rata-rata terendah:")
print(bottom_restaurants)

# Menampilkan seluruh dataset dengan rating rata-rata
print("Seluruh dataset dengan rating rata-rata:")
print(average_ratings)