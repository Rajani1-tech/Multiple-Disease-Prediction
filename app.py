import streamlit as st
from user import login, sign_up
from streamlit_option_menu import option_menu
from app_diabetes import app_diabetes
from app_heart import app_heartdisease, model
from app_breast_cancer import app_breast_cancer
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
            'Model',
            'Heart Disease Model Test Result',
            'Diabetes Model Test Result',
            'EDA for Heart Disease',
            'EDA for Diabetes Disease'
        ],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'clock', 'database', 'bar-chart', 'bar-chart', 'graph-up', 'graph-up'],
        default_index=1
    )

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
