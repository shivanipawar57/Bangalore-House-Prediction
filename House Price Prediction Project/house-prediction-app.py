import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image



html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">House Price Prediction App</h2>
    </div>
    """

html_temp2 = """
    <div style="background-color:none;padding:10px">
    <h6 style="text-align:center;">This app predicts house price</h6>
    </div>
"""

st.markdown(html_temp, unsafe_allow_html=True)
st.markdown(html_temp2, unsafe_allow_html=True)


@st.cache_resource
def pickle_download():
    # Reads in saved classification model
    model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))
    return model

model = pickle_download()

@st.cache_resource
def download():
    X = pd.read_csv('X.csv')
    return X

X = download()


def turn_off():
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]


Location = st.sidebar.selectbox('Location', tuple(X.columns[3:]), on_change=turn_off)
BHK = int(st.sidebar.selectbox('BHK', tuple(range(1,17)), on_change=turn_off))
Bath = int(st.sidebar.selectbox('Bath', tuple(range(1,17)), on_change=turn_off))
Area = st.sidebar.slider('Area', 300, 30000, 1500, on_change=turn_off)


if "button1" not in st.session_state:
    st.session_state.button1 = False

def callback():
    st.session_state.button1 = True
    
col1, col2 = st.columns([2,6])

with col1:
    button1 = st.button("Predict Price",on_click=callback)
    
    if button1 or st.session_state.button1:

        def predict_price(location,sqft,bath,bhk):    
            loc_index = np.where(X.columns==location)[0][0]

            x = np.zeros(len(X.columns))
            x[0] = sqft
            x[1] = bath
            x[2] = bhk
            if loc_index >= 0:
                x[loc_index] = 1

            return model.predict([x])[0]

        result = round(predict_price(Location, Area, Bath, BHK),2)
        st.header(str(result) +' Lacs approx')
    
with col2:
    image = Image.open('house.jpeg')
    st.image(image)




    

