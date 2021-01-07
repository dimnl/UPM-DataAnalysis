import streamlit as st
import numpy as np
import pandas as pd

import altair as alt
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def main():

    df = load_data()

    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])

    if page == 'Homepage':
        st.title('Predicting Relevancy of Colonoscopy frames')
        st.header('Problem definition')
        st.markdown("""
        The Health Analytics department from a very well-known hospital located in a European
        country has started a project to improve the results of the diagnosis related to colonoscopies.
        The project consists of highlighting certain areas of video frames of a colonoscopy that might
        be considered more relevant for colorectal illnesses due to the texture of the image in them.
        
        To achieve this, the department has extracted some information about the encoded video
        stream in blocks of 64x64 pixels and they have labelled each block as relevant or not relevant
        for performing a thorough examination of that part of the frame. It has been possible to obtain
        26 features from 16,000 different blocks with their associated class labels.
        
        The system to be developed should aim to classify a frame block as relevant or not relevant in
        the colonoscopy. We are also asked to identify the most important variables that influence in a
        block for being relevant or not as, well as any other useful information from the analysis of the
        dataset.
        """)
        
        st.markdown('')
           
    elif page == 'Exploration':
        st.title('Explore the Dataset and its characteristics')
        st.markdown('The dataset is characterized by 26 features and 1 labeled column if the frame is relevant or not.')

        if st.checkbox('Show description of variables'):
            st.markdown("""
            **quality**: a measure of the quality of the recorded video.
            
            **bits**: number of bits used to encode that block in the video stream.
            
            **intra_parts**: number sub-blocks inside this block that are not encoded by making use of
            information in other frames.
            
            **skip_parts**: number sub-blocks inside this block that are straight-forward copied from another
            frame.
            
            **inter_16x16_parts**: number of sub-blocks inside this block making use of information in other
            frames and whose size is 16x16 pixels.
            
            **inter_4x4_parts**: number of sub-blocks inside this block making use of information in other
            frames and whose size is 4x4 pixels.
            
            **inter_other_parts**: number of sub-blocks inside this block making use of information in other
            frames and whose size is different from 16x16 and 4x4 pixels.
            
            **non_zero_pixels**: number of pixels different from 0 after encoding the block.
            
            **frame_width**: the width of the video frame in pixels.
            
            **frame_height**: the height of the video frame in pixels.
            
            **movement_level**: a measure of the level of movement of this frame with respect the previous
            one.
            
            **mean**: mean of the pixels of the encoded block.
            
            **sub_mean_1**: mean of the pixels contained in the first 32x32 sub-bock of the current block.
            
            **sub_mean_2**: mean of the pixels contained in the second 32x32 sub-bock of the current block.
            
            **sub_mean_3**: mean of the pixels contained in the third 32x32 sub-bock of the current block.
            
            **sub_mean_4**: mean of the pixels contained in the fourth 32x32 sub-bock of the current block.
            
            **var_sub_blocks**: variance of the four previous values.
            
            **sobel_h**: mean of the pixels of the encoded block after applying the Sobel operator in
            horizontal direction.
            
            **sobel_v**: mean of the pixels of the encoded block after applying the Sobel operator in vertical
            direction.
            
            **variance**: variance of the pixels of the encoded block.block_movement_h: a measure of the movement of the current block in the horizontal
            direction.
            
            **block_movement_v**: a measure of the movement of the current block in the vertical direction.
            
            **var_movement_h**: a measure of the variance of the movements inside the current block in the
            horizontal direction.
            
            **var_movement_v**: a measure of the variance of the movements inside the current block in the
            vertical direction.
            
            **cost_1**: a measure of the cost of encoding this block without partitioning it.
            
            **cost_2**: a measure of the cost of encoding this block without partitioning it and without
            considering any movement in it.
            
            **relevant**: the target variable that indicates whether the current block is relevant (1) or not (0).
            """)
        
        if st.checkbox('Show sample dataset'):
            st.markdown('Some rows can be seen below to see what the data looks like:')
            st.dataframe(df.head())
        
        if st.checkbox('Show statistics per column'):
            st.dataframe(df.describe())
        
        st.header('Analysing column relations')

        corr_data = df.corr().stack().reset_index().rename(
            columns={0: 'Correlation', 'level_0': 'Feature 1', 'level_1': 'Feature 2'}
        )
        corr_chart = alt.Chart(corr_data).mark_rect().encode(
            x = 'Feature 1',
            y = 'Feature 2',
            color=alt.Color('Correlation:Q', scale=alt.Scale(domainMid=0,scheme="blueorange")),
            tooltip = [
                "Feature 1",
                "Feature 2",
                'Correlation'
            ]
        )

        st.altair_chart(corr_chart, use_container_width=True)

    else:
        st.title('Modelling')
        model, accuracy = train_model(df)
        st.write('Accuracy: ' + str(accuracy))
        st.header('Make prediction')
        row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)

        st.markdown('#### Predicted')
        st.text(model.predict(df.drop('relevant', axis=1).loc[row_number].values.reshape(1,-1))[0])

        st.markdown('#### Actual value')
        st.text(df.relevant.loc[row_number])

        st.header('Inspect feature importances')
        rf_importance = model.feature_importances_
        feature_names = df.drop('relevant', axis=1).columns
        
        features_to_display = st.slider(label="Amount of features to display", min_value=1, max_value=26, value=10, step=1)

        sorted_idx = rf_importance.argsort()[-features_to_display:]

        feature_df = pd.DataFrame({'Importances': rf_importance[sorted_idx], 'Feature names': feature_names[sorted_idx]})
        feature_chart = alt.Chart(feature_df).mark_bar().encode(
            x='Importances',
            y=alt.Y('Feature names:N', sort='-x'),
            tooltip=[
                alt.Tooltip("Feature names", title="Feature name"),
                alt.Tooltip("Importances", title="Importance")
            ]
        )
        st.altair_chart(feature_chart, use_container_width=True)

@st.cache
def split_and_normalize_df(df):
    X = df.drop('relevant', axis=1)
    y = df.relevant

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train_norm = normalize(X_train, norm='l2')
    X_test_norm = normalize(X_test, norm='l2')

    return X_train_norm, y_train, X_test_norm, y_test

@st.cache(allow_output_mutation=True)
def train_model(df):
    X_train_norm, y_train, X_test_norm, y_test = split_and_normalize_df(df)
    model = RandomForestClassifier(random_state=1, n_jobs=-1)
    model.fit(X_train_norm, y_train)

    return model, model.score(X_test_norm, y_test)

@st.cache
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/dimnl/UPM-DataAnalysis/main/data_prep.csv")


if __name__ == '__main__':
    main()