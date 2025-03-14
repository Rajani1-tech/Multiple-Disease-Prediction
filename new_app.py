import streamlit as st
import sqlite3
from streamlit_option_menu import option_menu
from app_diabetes import app_diabetes
from app_heart import app_heartdisease, model
from app_breast_cancer import app_breast_cancer

# Set up the page
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Connect to SQLite database
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

# Function to create users table if it doesn't exist
def create_users_table():
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL)''')
    conn.commit()

create_users_table()

# Function to check user credentials
def check_login(username, password):
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone() is not None

# Function to register a new user
def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:  # Catch duplicate username errors
        return False

# Session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login or Signup Page
if not st.session_state.logged_in:
    st.title("🔒 Login / Sign Up")
    menu_choice = st.radio("Select an option:", ["Login", "Sign Up"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if menu_choice == "Login":
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

    elif menu_choice == "Sign Up":
        if st.button("Sign Up"):
            if register_user(username, password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists")

else:
    # Show the main application after login
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',
                               ['Diabetes Prediction', 'Heart Disease Prediction', 'Breast Cancer Prediction'],
                               menu_icon='hospital-fill',
                               icons=['activity', 'heart', 'person'],
                               default_index=1)

    if selected == 'Diabetes Prediction':
        app_diabetes()
    elif selected == 'Heart Disease Prediction':
        app_heartdisease(model)
    elif selected == 'Breast Cancer Prediction':
        app_breast_cancer()
