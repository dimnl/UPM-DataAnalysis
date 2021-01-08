# UPM-DataAnalysis

## Scope

The scope of the project is to analyse colonoscopy data, and identify the most influential variables. Moreover, the scope is also to solve the classification task of predicting the relevant and not relevant frames. We aim to aid the experts to achieve a more precise analysis of a colonoscopy video, by highlighting blocks which would prove more relevant to the diagnosis.

The project also aims to deploy the model which was evaluated the most fitting, in a system, which domain experts (such as doctors) can easily use. This deployed system would also provide interactive visualizations of the data and any other relevant information. 

### Definition colonoscopy

Colonoscopy is the examination process of the digestive system with a camera. This process helps in visually diagnosing different digestive system related diseases.

## Deployed product

The product is deployed using Heroku (free tier) on [this link](https://data-analysis-upm.herokuapp.com/).

## Install and run locally

- Clone this repository
- In terminal, run the following command (assuming no conda environment) to install the necessary packages: `pip install -r requirements.txt`
- To then start the instance locally, run the following command in terminal: `streamlit run app.py`

## Description files

- **data/data.csv** is the original dataset as obtained from the assignment
- **data/data\_prep.csv** is cleaned dataset obtained from _UPM-Colonoscopy.ipynb_
- **.gitignore** to prevent local/unneeded files to end up in the GitHub
- **Procfile** to let Heroku know what command to run to setup deployment (``setup.sh` and `streamlit run app.py`)
- **setup.sh** to initialize streamlit configuration correctly for on Heroku
- **requirements.txt** to have all required libraries with each specific version that is used; used in particular to install the necessary libraries on the server. This file is obtained by using the _pipreqs_ library in python and running `pipreqs .` in the _UPM-DataAnalysis_ directory.
- **app.py** is the main python file used in deployment. This file is run on the server to calculate the back-end as well as display the front-end. It uses the libraries as specified in _requirements.txt_
- **UPM-Colonoscopy.ipynb** is the jupyter notebook in which initial data exploration was performed; after that part of this was adopted and rewritten to use in _app.py_. Furthermore, in here the data was cleaned to get _data\_prep.csv_
- **UPM-Models.ipynb** is the jupyter notebook in which different models (Random Forest, KNN and SVM) were trained, compared and optimized. Part of the evaluation was adopted and rewritten to use in _app.py_. The best model (Random Forest) was furthermore saved to get _rfc.sav_ using _pickle_
- **rfc.sav** is the saved Random Forest Classifier, which is used to load the model in _app.py_ and prevent unnecessary computation every time the deployed product is reloaded.

