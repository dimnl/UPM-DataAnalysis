import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def main():

    df = load_data()

    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])

    if page == 'Homepage':
        st.title('Predicting Relevancy of Colonoscopy frames')
        st.text('The dataset is characterized by 26 features and 1 labeled column if the frame is relevant or not')
        st.dataframe(df.head())
    elif page == 'Exploration':
        st.title('Explore the Dataset and its characteristics')
        if st.checkbox('Show column descriptions'):
            st.dataframe(df.describe())
        
        st.markdown('### Analysing column relations')
        st.text('Correlations:')
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot()

    else:
        st.title('Modelling')
        model, accuracy = train_model(df)
        st.write('Accuracy: ' + str(accuracy))
        st.markdown('### Make prediction')
        st.dataframe(df)
        row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
        st.markdown('#### Predicted')
        st.text(model.predict(df.drop('relevant', axis=1).loc[row_number]))


@st.cache
def train_model(df):
    X = df.drop('relevant', axis=1)
    y = df.relevant

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train_norm = normalize(X_train, norm='l2')
    X_test_norm = normalize(X_test, norm='l2')
    
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X_train_norm, y_train)

    return model, model.score(X_test_norm, y_test)

@st.cache
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/dimnl/UPM-DataAnalysis/main/data_prep.csv")


if __name__ == '__main__':
    main()