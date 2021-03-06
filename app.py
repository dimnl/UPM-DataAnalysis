import streamlit as st
import numpy as np
import pandas as pd
import base64
import pickle
from urllib.request import urlopen

import altair as alt
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

def main():
    df = load_data()
    dirty_df = load_dirty_data()
    feature_names = df.drop('relevant', axis=1).columns
    st.sidebar.title("UPM Data Analysis")
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'PCA', 'Prediction'])
    
    st.sidebar.markdown(
        """
        **Important notice**
        
        This is a fully self-contained application and it may take a while the first time each page is computed.
        The results are cached, however, and there is no need to wait that long again after this first calculation.
        """
    )
    st.sidebar.markdown(
        """
        **Additional information**
        
        All  plots are interactive, you can point your mouse on the graphs' data points for additional information.
        Furthermore, you could also zoom in and move around in line graphs.

        Source code can be found at [GitHub](https://github.com/dimnl/UPM-DataAnalysis).
        """
    )
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

        st.header("Colonoscopy definition")
        st.markdown("""
        Colonoscopy is the examination process of the digestive system with a camera. 
        This process helps in visually diagnosing different digestive system related diseases.
        """)
           
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
            st.dataframe(dirty_df.head())
        
        if st.checkbox('Show statistics per column'):
            st.dataframe(dirty_df.describe())

        if st.checkbox('Show null values per column'):
            st.table(dirty_df.isnull().sum().sort_values(ascending=False))

        st.header('Cleaning dataset')
        st.markdown("The sub_mean_3 can be calculated from the other sum_means and the mean")
        st.code('''
        sub_mean_3 = 4*mean - sub_mean_1 - sub_mean_2 - sub_mean_4
        ''', language='python')


        st.markdown("""
        The cost_1 and cost_2 are highly correlated (as is also shown below) and have near equal values.
        Therefore, it is chosen to impute the cost_2 values with cost_1 values.
        """)

        st.markdown("Lastly, as there are only two missing values from relevant, we can simply drop these rows.")        

        
        st.header('Analysing column relations')

        corr_data = calc_correlation(df)

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

        if st.checkbox("Show feature correlations in more detail"):
            st.subheader("Correlation in detail")
            st.markdown("""
            Which two features would you like to see in more detail?
            """)
    
            feature_1 = st.selectbox("First feature", options=feature_names, index=24)
            feature_2 = st.selectbox("Second feature", options=feature_names, index=25)
            
            line_chart_feature = alt.Chart(df).mark_circle(size=15).encode(
                x = feature_1,
                y = feature_2,
                color = alt.Color('relevant:O', scale=alt.Scale(scheme="tableau10")),
                tooltip=[feature_1, feature_2, 'relevant']
            ).interactive()
    
            st.altair_chart(line_chart_feature, use_container_width=True)
    
            st.markdown("""
            Also note that the scatterplot is interactive, meaning that you could zoom in and move around. 
            """)
        
    elif page == 'PCA':
        st.title('Principal Component Analysis')
        st.header('2 Dimensional')
        st.markdown("With MinMax Normalization")
        df_pca, _ = pca_2d(df)
        pca_2d_chart = alt.Chart(df_pca.join(df.relevant)).mark_circle(size=15).encode(
                x = 'x',
                y = 'y',
                color = alt.Color('relevant:O', scale=alt.Scale(scheme="tableau10")),
                tooltip=['x', 'y', 'relevant']
            ).interactive()
        
        st.altair_chart(pca_2d_chart, use_container_width=True)

        st.header('3 Dimensional')
        st.markdown("With MinMax Normalization")
        df_pca, pca = pca_3d(df)
        
        pca_3d_xy_chart = alt.Chart(df_pca.join(df.relevant)).mark_circle(size=15).encode(
            x = 'x',
            y = 'y',
            color = alt.Color('relevant:O', scale=alt.Scale(scheme="tableau10")),
            tooltip=['x', 'y', 'relevant']
        ).interactive()

        st.altair_chart(pca_3d_xy_chart, use_container_width=True)

        pca_3d_xz_chart = alt.Chart(df_pca.join(df.relevant)).mark_circle(size=15).encode(
            x = 'x',
            y = 'z',
            color = alt.Color('relevant:O', scale=alt.Scale(scheme="tableau10")),
            tooltip=['x', 'z', 'relevant']
        ).interactive()

        st.altair_chart(pca_3d_xz_chart, use_container_width=True)

        pca_3d_yz_chart = alt.Chart(df_pca.join(df.relevant)).mark_circle(size=15).encode(
            x = 'y',
            y = 'z',
            color = alt.Color('relevant:O', scale=alt.Scale(scheme="tableau10")),
            tooltip=['y', 'z', 'relevant']
        ).interactive()

        st.altair_chart(pca_3d_yz_chart, use_container_width=True)

        st.subheader('Feature importances')
        features_to_display = st.slider(label="Amount of features to display", min_value=1, max_value=26, value=10, step=1)
        comp = abs(pca.components_)
        sorted_idx = comp[0].argsort()[-features_to_display:]

        st.markdown("First component")
        feature1_df = pd.DataFrame({'Importances': comp[0][sorted_idx], 'Feature names': feature_names[sorted_idx]})
        feature1_chart = alt.Chart(feature1_df).mark_bar().encode(
            x='Importances',
            y=alt.Y('Feature names:N', sort='-x'),
            tooltip=[
                alt.Tooltip("Feature names", title="Feature name"),
                alt.Tooltip("Importances", title="Importance")
            ]
        )
        st.altair_chart(feature1_chart, use_container_width=True)

        st.markdown("Second component")
        sorted_idx = comp[1].argsort()[-features_to_display:]
        feature2_df = pd.DataFrame({'Importances': comp[1][sorted_idx], 'Feature names': feature_names[sorted_idx]})
        feature2_chart = alt.Chart(feature2_df).mark_bar().encode(
            x='Importances',
            y=alt.Y('Feature names:N', sort='-x'),
            tooltip=[
                alt.Tooltip("Feature names", title="Feature name"),
                alt.Tooltip("Importances", title="Importance")
            ]
        )
        st.altair_chart(feature2_chart, use_container_width=True)

        st.markdown("Third component")
        sorted_idx = comp[2].argsort()[-features_to_display:]
        feature3_df = pd.DataFrame({'Importances': comp[2][sorted_idx], 'Feature names': feature_names[sorted_idx]})
        feature3_chart = alt.Chart(feature3_df).mark_bar().encode(
            x='Importances',
            y=alt.Y('Feature names:N', sort='-x'),
            tooltip=[
                alt.Tooltip("Feature names", title="Feature name"),
                alt.Tooltip("Importances", title="Importance")
            ]
        )
        st.altair_chart(feature3_chart, use_container_width=True)

                
    else:
        st.title('Modelling')
        btn_retrain = st.button('Retrain model (takes some time!)')
        if btn_retrain:
            model, accuracy = train_model(df)
            st.write('Accuracy of trained model (Random Forest): ' + str(accuracy))
        else:
            model = load_model()

        st.header('Make prediction')

        st.markdown("""
        Upload a file to make a prediction, following the structure of the data as defined in Data Exploration. 
        The 'relevant' column is not used and does not have to be included; this value is predicted for the uploaded data.
        It should be a CSV file and be cleaned already, i.e. not containing empty (or `null`) values.
        """)
        
        uploaded_file = st.file_uploader('Choose a file to predict on')
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)

            if st.checkbox('Show uploaded file'):
                st.dataframe(df_uploaded)
            
            if st.checkbox('Predict single sample'):
                row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)

                st.write('Predicted relevant: ' + str(int(model.predict(df.drop('relevant', axis=1).loc[row_number].values.reshape(1,-1))[0])))
                st.write('Actual relevant: ' + str(int(df.relevant.loc[row_number])))

            if st.checkbox('Predict for whole dataset'):
                df_predicted = predict_for_uploaded_df(model, df_uploaded)
                st.dataframe(df_predicted)

                csv, b64 = generate_download_link(df_predicted)
               
                st.markdown(
                    f"""
                    <a href="data:file/csv;base64, {b64}" download="predicted.csv">
                    <input type="button" value="Download predicted csv">
                    </a>
                    """,
                    unsafe_allow_html=True
                )

        st.header('Inspect feature importances')
        rf_importance = model.feature_importances_
                
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

        st.subheader("Permutation importances")
        if btn_retrain:
            _, _, X_test, y_test = split_and_normalize_df(df)
            perm_df = calc_permutation_importance(model, X_test, y_test, feature_names)
            perm_chart = alt.Chart(perm_df).mark_bar().encode(
                x='Importances',
                y=alt.Y('Feature names:N', sort='-x'),
                tooltip=[
                    alt.Tooltip("Feature names", title="Feature name"),
                    alt.Tooltip("Importances", title="Importance")
                ]
            )
            st.altair_chart(perm_chart, use_container_width=True)
        else:
            st.markdown("""
            Unfortunately the inspection of the permutaion importances only functions with a trained model from scratch.
            This is a downside of loading the pre-trained model as we cannot inspect the model that thoroughly with sklearn's inspections library.
            Please select the retrain button on top to also see these.
            """)

def minmax_scale(df):
    df_train = df.drop('relevant', axis=1)
    scaler = MinMaxScaler()
    df_train = scaler.fit_transform(df_train)
    return df_train

@st.cache
def calc_correlation(df):
    corr_data = df.corr().stack().reset_index().rename(
        columns={0: 'Correlation', 'level_0': 'Feature 1', 'level_1': 'Feature 2'}
    )
    return corr_data

@st.cache
def pca_2d(df):
    df_train = minmax_scale(df)
    
    pca = PCA(n_components=2)
    pca_comps = pca.fit_transform(df_train)
    df_pca = pd.DataFrame(data=pca_comps, columns=['x', 'y'])
    return df_pca, pca

@st.cache
def pca_3d(df):
    df_train = minmax_scale(df)
    
    pca = PCA(n_components=3)
    pca_comps = pca.fit_transform(df_train)
    df_pca = pd.DataFrame(data=pca_comps, columns=['x', 'y', 'z'])
    return df_pca, pca

def split_and_normalize_df(df):
    X = df.drop('relevant', axis=1)
    y = df.relevant

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train_norm = normalize(X_train, norm='max')
    X_test_norm = normalize(X_test, norm='max')

    return X_train_norm, y_train, X_test_norm, y_test

@st.cache(allow_output_mutation=True)
def train_model(df):
    X_train_norm, y_train, X_test_norm, y_test = split_and_normalize_df(df)
    model = RandomForestClassifier(random_state=1, n_jobs=-1)
    model.fit(X_train_norm, y_train)

    return model, model.score(X_test_norm, y_test)

def predict_for_uploaded_df(clf, uploaded_df):
    if 'relevant' in uploaded_df.columns:
        uploaded_df = uploaded_df.drop('relevant', axis=1)
    predictions = clf.predict(uploaded_df)
    return uploaded_df.join(pd.DataFrame({'pred_relevant': predictions}))

@st.cache
def calc_permutation_importance(clf, X_test, y_test, feature_names):
    perm_importance =  permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=1)
    sorted_idx = perm_importance.importances_mean.argsort()
        
    perm_df = pd.DataFrame({'Importances': perm_importance.importances_mean[sorted_idx].T, 'Feature names': feature_names[sorted_idx]})
    return perm_df

@st.cache    
def generate_download_link(df):
    # Some encoding/decoding magic to create download
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return csv, b64

@st.cache(allow_output_mutation=True)
def load_model():
    url = "https://raw.githubusercontent.com/dimnl/UPM-DataAnalysis/main/rfc.sav"
    with urlopen(url) as f:
        loaded_model = pickle.load(f)
    return loaded_model

@st.cache
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/dimnl/UPM-DataAnalysis/main/data/data_prep.csv")
    
@st.cache
def load_dirty_data():
    return pd.read_csv("https://raw.githubusercontent.com/dimnl/UPM-DataAnalysis/main/data/data.csv")
    
if __name__ == '__main__':
    main()