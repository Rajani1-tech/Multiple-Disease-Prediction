import streamlit as st
import streamlit_authenticator as stauth
import re
import sqlite3
from database import create_users_table  # Import the create_users_table function
import hashlib  # Import hashlib for password hashing

# Ensure the table exists when the app starts
create_users_table()

# Database connection and functions
def create_connection():
    return sqlite3.connect('new_user.db')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def insert_user(email, username, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)", (email, username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_emails():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users")
    emails = [row[0] for row in cursor.fetchall()]
    conn.close()
    return emails

def get_usernames():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users")
    usernames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return usernames

def get_user(email):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

# Form validation functions
def validate_email(email):
    pattern = r"^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"
    return bool(re.match(pattern, email))

def validate_username(username):
    pattern = "^[a-zA-Z0-9]*$"
    return bool(re.match(pattern, username))

# Sign up function
def sign_up():
    """Sign up form"""
    with st.form(key='signup', clear_on_submit=True):
        st.subheader(':green[Sign Up]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        username = st.text_input(':blue[Username]', placeholder='Enter Your Username')
        password1 = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')
        password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')

        # Form validation on submit
        if st.form_submit_button('Sign Up'):
            if email and username and password1 and password2:
                if validate_email(email):
                    if email not in get_user_emails():
                        if validate_username(username):
                            if username not in get_usernames():
                                if len(username) >= 2:
                                    if len(password1) >= 6:
                                        if password1 == password2:
                                            hashed_password = hash_password(password2)  # Hash password before storing
                                            success = insert_user(email, username, hashed_password)
                                            if success:
                                                st.success('Account created successfully!!')
                                                st.balloons()
                                            else:
                                                st.error('Error creating account. Try again.')
                                        else:
                                            st.warning('Passwords Do Not Match')
                                    else:
                                        st.warning('Password is too Short')
                                else:
                                    st.warning('Username Too Short')
                            else:
                                st.warning('Username Already Exists')
                        else:
                            st.warning('Invalid Username')
                    else:
                        st.warning('Email Already Exists!!')
                else:
                    st.warning('Invalid Email')

# Login function
def login():
    """Login Form"""
    with st.form(key='login'):
        st.subheader(':green[Login]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        password = st.text_input(':blue[Password]', placeholder='Enter Your Password', type='password')

        if st.form_submit_button('Login'):
            if email and password:
                user = get_user(email)
                if user:
                    stored_password = user[2]  # Password is in the third column
                    if hash_password(password) == stored_password:  # Compare hashed passwords
                        st.success('Login successful!')
                    else:
                        st.error('Incorrect password')
                else:
                    st.error('User not found')

# Main function to toggle between Sign Up and Login
def main():
    """Main Function to Display SignUp/Login"""
    st.title('Welcome to the Authentication System')

    # Option to switch between login or signup
    choice = st.radio("Choose an option", ['Sign Up', 'Login'])

    if choice == 'Sign Up':
        sign_up()
    elif choice == 'Login':
        login()

if __name__ == "__main__":
    main()
