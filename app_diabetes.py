import streamlit as st
import pickle

def app_diabetes():
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    test_accuracy = 0.753246
    diabetes_model = pickle.load(open('saved_models/diabetes_model_decision_tree.sav', 'rb'))
   

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic using Decision Tree Model. The test accuracy of the model is ' + str(test_accuracy)  
        else:
            diab_diagnosis = 'The person is not diabetic Decision Tree Model. The test accuracy of the model is ' + str(test_accuracy)

    st.success(diab_diagnosis)

    if st.button('Show Model Performance'):
        st.subheader("Model Performance on Test Data")


#     # Accuracy Score
#         accuracy = accuracy_score(y_test, y_pred)
#         st.write(f"**  Model Accuracy:** {accuracy:.2f}")

#     # Classification Report
#         st.write(" **Classification Report:**")

#     #   Convert the classification report into a DataFrame
#         report_dict = classification_report(y_test, y_pred, output_dict=True)

#    # Extract relevant data
#         report_df = pd.DataFrame(report_dict).transpose()
#         report_df = report_df.rename(index={'0': 'No Disease (0)', '1': 'Disease (1)'})

#    # Round values for better readability
#         report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)

#    # Display as a Streamlit table
#         st.dataframe(report_df)



#     # Confusion Matrix
#         st.write(" **Confusion Matrix:**")
#         conf_matrix = confusion_matrix(y_test, y_pred)

#     # Plot Confusion Matrix
#         fig, ax = plt.subplots(figsize=(5, 3))
#         sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         st.pyplot(fig)