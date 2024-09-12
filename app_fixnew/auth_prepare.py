import yaml
from yaml.loader import SafeLoader
import os 
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
import pandas as pd
import streamlit as st

ratings = pd.read_csv('data/dataset/ratings_restaurant.csv')

ratings['user_id'] = ratings['user_id'].astype(str)
# Fungsi untuk memuat atau menyimpan konfigurasi yaml
CONFIG_PATH = 'config.yaml'

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as file:
            return yaml.load(file, Loader=SafeLoader)
    else:
        return {'credentials': {'usernames': {}}, 'cookie': {'expiry_days': 30, 'key': 'random_key', 'name': 'auth_cookie'}, 'preauthorized': {}}
def save_config(config):
    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

if not os.path.exists(CONFIG_PATH):
    # Buat daftar pengguna untuk authenticator
    users = ratings['user_id'].unique().tolist()

    # Buat password yang sama dengan user_id
    hashed_passwords = [Hasher([user]).generate()[0] for user in users]

    # Buat konfigurasi untuk authenticator
    config = {
        'credentials': {
            'usernames': {
                user: {'name': user, 'password': pwd} for user, pwd in zip(users, hashed_passwords)
            }
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'random_key',
            'name': 'auth_cookie'
        },
        'preauthorized': {}
    }

    # Simpan konfigurasi ke file yaml sementara
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    st.write('data auth/config.yaml sudah berhasil dibuat')
else:
    st.write('data uath/config.yaml sudah ada')




config = load_config()


# Menambah admin jika belum ada
if 'admin' not in config['credentials']['usernames']:
    admin_password = Hasher(['admin']).generate()[0]
    config['credentials']['usernames']['admin'] = {
        'name': 'admin',
        'password': admin_password
    }
    save_config(config)
    st.write('admin sudah ditambahkan')
else:
    st.write('admin sudah ada')

