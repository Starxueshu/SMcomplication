# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("Machine learning-based prediction for postoperative complication among patients with metastatic spinal disease")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Currentsmoking = st.sidebar.selectbox("Current smoking", ("No", "Yes"))
Diabetes = st.sidebar.selectbox("Diabetes", ("No", "Yes"))
Hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
Viscerametastases = st.sidebar.selectbox("Visceral metastases", ("No", "Yes"))
Bloodtransfusion = st.sidebar.selectbox("Blood transfusion", ("No", "Yes"))
Surgicalsegements= st.sidebar.selectbox("Surgical segments", ("One", "Two", "≥Three"))
Surgicaltime= st.sidebar.slider("Surgery time (min)", 150, 400)

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_roundAI.pkl")
    x = pd.DataFrame([[Currentsmoking, Diabetes, Hypertension, Viscerametastases, Bloodtransfusion, Surgicalsegements,Surgicaltime]],
                     columns=["Currentsmoking","Diabetes","Hypertension","Viscerametastases","Bloodtransfusion","Surgicalsegements","Surgicaltime"])

    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["One", "Two", "≥Three"], [1, 2, 3])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Postoperative complication risk: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.50:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_trainy = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ["Currentsmoking","Diabetes","Hypertension","Viscerametastases","Bloodtransfusion","Surgicalsegements","Surgicaltime"]]
    y_train = y_trainy.Complication
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    #st.text(shap_value)

    shap.initjs()
    #image = shap.plots.force(shap_value)
    #image = shap.plots.bar(shap_value)

    shap.plots.waterfall(shap_value[0])
    st.pyplot(bbox_inches='tight')
    st.set_option('deprecation.showPyplotGlobalUse', False)


st.subheader('About the model')
st.markdown('The complimentary online calculator utilizes the advanced extreme gradient boosting machine algorithm and has demonstrated exceptional performance, boasting an AUC of 0.924 (95% CI: 0.884-0.965) during validation. However, it is crucial to emphasize that this model is intended solely for research purposes. Consequently, clinical treatment decisions regarding metastatic spinal disease should not rely exclusively on this AI platform. Instead, its predictions should be considered as an auxiliary resource to aid decision-making, complementing the expertise and judgment of healthcare professionals.')
