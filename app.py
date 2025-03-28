import streamlit as st
from user import login, sign_up, check_recent_predictions
from streamlit_option_menu import option_menu
from app_diabetes import app_diabetes
from app_heart import app_heartdisease, model, show_heart_model_test_result, show_eda_for_heart_disease, show_logistic_regression_description
from user import get_user_predictions  # Import database function

# Set page config at the top before any other Streamlit command
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def load_health_assistant():
    """Health Assistant Page"""
    email = st.session_state.get('user_email', 'Guest')  # Get logged-in user's email
    
    with st.sidebar:
      selected = option_menu(
        'Multiple Disease Prediction System',
        [
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'My Predictions',
            'Heart Model Test Result',
            'Diabetes Model Test Result',
            'EDA for Heart Disease',
            'EDA for Diabetes Disease',
            'Model Description'
        ],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'clock', 'bar-chart', 'bar-chart', 'graph-up', 'graph-up', 'database'],
        default_index=1
    )
      # Check for alert if the user has predicted heart or diabetes disease 3 times within a month
    if selected in ['Heart Disease Prediction', 'Diabetes Prediction']:
        disease_type = 'Heart Disease' if selected == 'Heart Disease Prediction' else 'Diabetes'
        if check_recent_predictions(email, disease_type):
            st.warning(f"⚠️ **Alert:** You have predicted {disease_type} 3 or more times in the last month. Please consult a nearby doctor.")
    

    if selected == 'Diabetes Prediction':
        app_diabetes()
    elif selected == 'Heart Disease Prediction':
        app_heartdisease(model)
    elif selected == 'My Predictions':
        st.subheader("📜 My Previous Predictions")
        predictions = get_user_predictions(email)
        if predictions:
            for disease, inputs, result, timestamp in predictions:
                st.write(f"🦠 **Disease:** {disease}")
                st.write(f"🕒 **Date:** {timestamp}")
                st.write(f"📊 **Inputs:** {inputs}")
                st.write(f"🖍 **Prediction:** {result}")
                st.markdown("---")
        else:
            st.info("No past predictions found.")
    elif selected == 'Heart Model Test Result':
        show_heart_model_test_result()
    elif selected ==   'EDA for Heart Disease':
        show_eda_for_heart_disease()
    elif selected ==   'Model Description':
        show_logistic_regression_description()


def main():
    """Main Function to Display SignUp/Login"""
    if not st.session_state.logged_in:
        st.title("Welcome to Multiple Disease Prediction System 🏥")
        
        choice = st.radio("Choose an option", ['Sign Up', 'Login'])

        if choice == 'Sign Up':
            sign_up()
        elif choice == 'Login':
            login()
    else:
        load_health_assistant()

if __name__ == "__main__":
    main()
