import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Pass or Fail Prediction", page_icon="🎓")

st.title("🎓 Pass or Fail Prediction")
st.write("Enter student details below to predict whether they will **Pass** or **Fail**.")

st.sidebar.header("Student Details")
study_hours = st.sidebar.slider("Study Hours (per day)", 0.0, 12.0, 5.0, 0.5)
previous_result = st.sidebar.slider("Previous Result (%)", 0, 100, 75)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)

st.write("### Input Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Study Hours", f"{study_hours}")
col2.metric("Previous Result", f"{previous_result}%")
col3.metric("Attendance", f"{attendance}%")

if st.button("🔮 Predict Result"):
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        input_data = scaler.transform([[study_hours, previous_result, attendance]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("🎉 Prediction: **PASS**")
            st.balloons()
        else:
            st.error("😞 Prediction: **FAIL**")
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Please run the notebook first to train and save the model.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub Repository](https://github.com/mdshafi-786/PASS-OR-FAIL-PREDECTION-Using-MACHINE-LEARNING-AND-PYTHON-)")
