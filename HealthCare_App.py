import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf

st.set_page_config(page_title="Healthcare Charges App",layout="centered")
st.title("Predictive Healthcare Charges Using Neural Networks")


@st.cache_resource
def load_model_and_transformer():
    model=tf.keras.models.load_model("insurance_model.h5",compile=False)

    with open("fitted_column_transformer_for_insurance.pkl","rb") as ct_file:
        transformer=pickle.load(ct_file)

    return model,transformer


def data_preprocessing(data,transformer):
    columns=["age","sex","bmi","children","smoker","region"]
    df=pd.DataFrame([data],columns=columns)
    preprocessed=transformer.transform(df)

    return preprocessed

age=st.number_input("Enter your age",min_value=18,max_value=64,value=25)
sex=st.selectbox("Select your sex",["male","female"])
bmi=st.number_input("Enter your body mass index(bmi)",min_value=15.96,max_value=53.13,value=20.0)
children=st.number_input("Enter the number of children you have",min_value=0.00,max_value=5.0,value=2.0)
smoker=st.selectbox("Are you a smoker",["yes","no"])
region=st.selectbox("Select your region of origin",["northwest","northeast","southeast","southwest"])


model,transformer=load_model_and_transformer()

if st.button("Predict care charges"):   
  data=[age,sex,bmi,children,smoker,region]
  data=data_preprocessing(data,transformer)

  prediction=model.predict(data)

  st.success(f"Your predicted charges are: ${float(prediction[0]):,.2f}")


