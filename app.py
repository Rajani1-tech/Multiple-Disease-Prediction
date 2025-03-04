import streamlit as st
from user import login, sign_up
from streamlit_option_menu import option_menu
from app_diabetes import app_diabetes
from app_heart import app_heartdisease, model
from app_breast_cancer import app_breast_cancer


# Set page config at the top before any other Streamlit command
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    

# Your database functions here...

def load_health_assistant():
    """Health Assistant Page"""
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

def main():
    """Main Function to Display SignUp/Login"""
    # Check if user is logged in
    if not  st.session_state.logged_in:
        st.title("welcome to Health Assistant 🏥")
        
        choice = st.radio("Choose an option", ['Sign Up', 'Login'])

        if choice == 'Sign Up':
            sign_up()
        elif choice == 'Login':
            login()
    else:
        load_health_assistant()        

if __name__ == "__main__":
    main()
