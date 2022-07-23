import sys
import re
import os
import random
import datetime

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

import numpy as np
import pandas as pd

import streamlit as st

import requests

import mlflow
import mlflow.sklearn

import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from DataGeneration import DataGeneration
from BoxDisplayer import BoxDisplayer

import time
from contextlib import contextmanager


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


@st.cache
def get_unique_items_dict():  # Get unique items dict as static
    df = pd.read_csv('data/unique_items_df.csv')
    return {column: list(
        df[column][df[column].notnull()]) for column in df.columns}


@st.cache
def get_columns_description_df():  # Get description df as static
    columns_description = pd.read_csv(
        'data/HomeCredit_columns_description_improved.csv',
        delimiter=';', index_col=[0])
    return columns_description


@st.cache
def get_global_df():  # Load database_columns as static
    global_df = pd.read_csv('data/global_train_data.csv', nrows=10000)
    #global_df = pd.read_pickle('data/global_train_data.pkl')
    return global_df


@st.cache
def get_X_df(global_df):  # Load database_columns as static
    X = global_df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    X_df = pd.DataFrame(X.values,
                        columns=X.columns,
                        index=global_df['SK_ID_CURR']).astype(dtype=X.dtypes)
    return X_df


@st.cache
def get_mlflow_model():  # Load database_columns as static
    # Load MLflow model
    pipeline = mlflow.sklearn.load_model('data/MLflow/mlflow_model/')
    return pipeline


@st.cache
def get_features_lut():
    features_lut = pd.read_csv('data/features_lut.csv', index_col=[0])
    return features_lut


@st.cache
def get_proba_prediction(clf, scaled_df):
    proba_prediction = clf.predict_proba(scaled_df)
    return proba_prediction


@st.cache
def transform_global_df(pipeline, df):
    # Get pipeline SimpleImputer and StandardScaler
    imputer = pipeline[0]
    scaler = pipeline[1]
    impute_data = imputer.transform(df)
    scaled_data = scaler.transform(impute_data)
    scaled_data_df = pd.DataFrame(
        scaled_data, columns=df.columns, index=df.index)
    return scaled_data_df


@st.cache
def get_tree_explainer(clf, scaled_df):
    explainer = shap.TreeExplainer(clf, scaled_df)
    return explainer


with timer("Instantiation process"):
    # Get unique items dict as static
    unique_items_dict = get_unique_items_dict()
    # Get columns description df as static
    cols_des_df = get_columns_description_df()
    # Get features LUT
    features_lut = get_features_lut()
    # Load database_columns as static
    global_df = get_global_df()
    customer_idx_list = global_df['SK_ID_CURR'].to_list()
    # Load mlflow model as static
    pipeline = get_mlflow_model()
    # Get X and y value from data
    X_df = get_X_df(global_df)
    # Transform X to have data for input of classifier
    scaled_data_df = transform_global_df(pipeline, X_df)
    loan_proba_list = get_proba_prediction(pipeline[3], scaled_data_df)[:, 0]
    # SHAP explainer values (NumPy array)
    explainer = get_tree_explainer(pipeline[3], scaled_data_df)


# Dict containing base names of database files
fetch_LUT = {
    'AT': ('application_train', 0),
    'BU': ('bureau', 0),
    'PA': ('previous_application', 1),
    'PCB': ('POS_CASH_balance', 1),
    'IP': ('installments_payments', 1),
    'CCB': ('credit_card_balance', 1)
}


def plot_histo_with_hline(values, hline_pos):
    fig, ax = plt.subplots(figsize=(10, 3))
    y, x, _ = ax.hist(values, bins=200)
    ax.axvline(hline_pos, color='red', linewidth=5)
    return fig


def request_prediction(model_uri, data):
    headers = {"Content-Type": "text/csv"}

    data_csv = data.to_csv()
    response = requests.request(
        method='POST', headers=headers, url=model_uri, data=data_csv)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code, response.text))

    return response.json()


def fetch_data(idx_curr, idx_list):
    if idx_curr in idx_list:
        # Create data generator object
        dg = DataGeneration(fetch_LUT)
        # Check if all idx_curr dataframes are already process
        files_already_processed = True
        for key in fetch_LUT:
            if not os.path.exists(
                    'data/temp/'+str(idx_curr)+'/'+fetch_LUT[key][0]+'.csv'):
                files_already_processed = False
        # If at least one is not process, process all of them and save them
        if not files_already_processed:
            dg.write_idx_data(idx_curr)
        # Load idx_curr dataframes
        dfs_dict = dg.get_idx_data(idx_curr)
        return dfs_dict
    else:
        return None


row0 = st.columns(1)

# Display dashboard header
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .1, 1, .1))

row0_1.title('Home Credit Default Risk')

with row0_2:
    st.write('')

row0_2.subheader(
    'A Streamlit web app by [Damien Dous](damien.dous@gmail.com)')

row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown(
        "**To begin, please enter current customer idx.** 👇")

# Display customer id selection panel
row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
no_idx_curr = True
with row2_1:
    idx_list = [name for name in os.listdir("data/temp/")
                if os.path.isdir("data/temp/"+name) and name.isdigit()]
    select_idx_curr = st.selectbox(
        "Select one of our sample customer profiles idx",
        idx_list)
    st.markdown("**or**")
    text = st.empty()
    text_idx_curr = text.text_input(
        "Input your customer idx that you want \
		to predict if loan is accepted or declined")
    if text_idx_curr != '' and text_idx_curr.isdigit() \
            and int(text_idx_curr) in customer_idx_list:
        idx_curr = int(text_idx_curr)
    else:
        idx_curr = int(select_idx_curr)

    if text_idx_curr != '' and text_idx_curr.isdigit() and \
            int(text_idx_curr) not in customer_idx_list:
        st.markdown("custumer idx does not exist, it will display " +
                    str(idx_curr)+" custumer idx")

# Display predict button
select_btn = st.button('Select')

# When predict button is selected:
if ('idx_curr' in st.session_state and
        st.session_state['idx_curr'] == idx_curr) or select_btn:

    st.session_state['idx_curr'] = idx_curr

    # Separate dashboard in two columns
    row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
        (.1, 1, .1, 1, .1))
    with row3_1:
        st.subheader('Values that can be modifiable')

    with row3_2:
        st.subheader('Values that can not be modifiable')

    # Load data of the current idx
    dfs_dict = fetch_data(idx_curr,
                          global_df['SK_ID_CURR'].to_list())
    # Copy dataframes dict to not modified them directly
    dataframes_dict = {key: dfs_dict[key].copy()
                       for key in dfs_dict}
    # Create SHAP explainer dataframe for current data with initial data
    shap_values = explainer.shap_values(scaled_data_df.loc[idx_curr, :])
    shap_values_df = pd.DataFrame(shap_values,
                                  index=scaled_data_df.columns,
                                  columns=['Value'])
    shap_values_df['Percentage'] = shap_values_df['Value'].abs() / \
        shap_values_df['Value'].abs().sum()

    # Sort SHAP dataframe
    shap_sorted_features = list(
        shap_values_df['Value'].abs().sort_values(ascending=False).index)
    shap_sorted_values_df = shap_values_df.loc[shap_sorted_features]

    st.write('')

    editable_counter = 0
    counter = 0
    while editable_counter < 10:
        db_feature = shap_sorted_values_df.index[counter]
        # Check if feature is in features LUT
        if db_feature not in features_lut.index:
            print('ERROR : '+db_feature+' feature not in index')
            counter += 1
            continue
        feature_info = features_lut.loc[db_feature, :]
        # Check if feature has correspondance in LUT
        if pd.isnull(feature_info['TableName']):
            print('WARNING : '+db_feature+' not refind in LUT')
            counter += 1
            continue
        # Do not display ***_balance file value
        # because it contains more than one line for a customer
        if feature_info['TableName'] in ['POS_CASH_balance',
                                         'credit_card_balance',
                                         'bureau_balance']:
            print('WARNING : '+db_feature+' feature in balance tables')
            counter += 1
            continue

        # Get column description row for the considered feature
        column_description = cols_des_df.loc[(
            cols_des_df['Row'] == feature_info['Row']) & (
            cols_des_df['Table'] == feature_info['TableName']+'.csv'), :]
        # Process description for the considered feature
        description_text = column_description['Description Small'].iloc[0]\
            + ' - SHAP coef :' + \
            '{:.3f}'.format(shap_sorted_values_df.loc[db_feature, 'Value'])
        # Get type for the considered feature
        type_ = column_description['Type'].iloc[0]

        # Dispatch feature according it's considered fixed or editable
        if column_description['State'].iloc[0] == 'EDITABLE':
            # Display fixed feature on left
            box_displayer = BoxDisplayer(unique_items_dict, row3_1)
            box_displayer.display_editable_feature(
                feature_info['TableName'],
                dataframes_dict[feature_info['TableName']],
                feature_info['Row'],
                description_text,
                type_)
            editable_counter += 1
        elif column_description['State'].iloc[0] == 'FIXED':
            # Display fixed feature on left
            box_displayer = BoxDisplayer(unique_items_dict, row3_2)
            box_displayer.display_fixed_feature(
                feature_info['TableName'],
                dataframes_dict[feature_info['TableName']],
                feature_info['Row'],
                description_text,
                type_)
        counter += 1

    # Add predict button
    predict_btn = st.button('Prédire')

    # When predict button is activate
    if predict_btn or ('predict' in st.session_state
                       and st.session_state['predict'] == idx_curr):
        st.session_state['predict'] = idx_curr

        # DATA GENERATION FOR CLASSIFIER
        # Create DataGenration object
        dg = DataGeneration(fetch_LUT)
        # Process data from current customer database
        customer_data = dg.process_database(dataframes_dict)
        # Fill missing data for missing columns
        missing_columns = list(set(global_df.columns) -
                               set(customer_data.columns))
        customer_data.loc[0, missing_columns] = global_df.loc[
            global_df['SK_ID_CURR'] == idx_curr, missing_columns].values[0]
        customer_data = customer_data[global_df.columns]
        #data.to_csv('data/data_for_classifier.csv', index=False)

        # PREDICTION FROM MODEL
        customer_X = customer_data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        # make a prediction request to mlflow server
        MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
        # CHECK SERVER IS RUNNING AND CHECK SERVER RETURN GOOD VALUES
        server_pred = request_prediction(MLFLOW_URI, customer_X)[0]

        col4_1, col4_2, col4_3 = st.columns((1, .2, 1))
        with col4_1:
            st.metric('Custumer client number',
                      idx_curr)

        row4_1, row4_2, row4_3 = st.columns((.7, 1, .7))
        with row4_2:
            # Plot gauge with result
            fig = go.Figure(go.Indicator(
                domain={'x': [0, 1], 'y': [0, 1]},
                value=server_pred[0],
                mode="gauge+number+delta",
                title={'text': "Score"},
                delta={'reference': 0.5},
                gauge={'axis': {'range': [None, 1]},
                       'steps': [
                    {'range': [0, 0.5], 'color': "lightgrey"},
                    {'range': [0.5, 1], 'color': "lightgreen"}
                ],
                    'threshold':
                    {'line':
                     {'color': "red", 'width': 8},
                     'thickness': 1,
                     'value': 0.5
                     }
                }))
            st.plotly_chart(fig)

            # Display the most important features with coefficient
            # plot the SHAP values for the output of the first instance
            # Create SHAP explainer dataframe for current data with initial
            # data
            st.subheader('SHAP values')

            ct = shap_sorted_values_df.iloc[4, 1]

            #print(shap_values_df.to_dict())
            #shap_dict = shap_values_df.to_dict()
            #shap_plot = shap.plots.bar(shap_dict)

            shap_plot = shap.force_plot(explainer.expected_value,
            							np.array(shap_values_df['Value']),
            							scaled_data_df.loc[idx_curr, :],
            							matplotlib=True,
            							show=False,
            							contribution_threshold=ct)
            st.pyplot(shap_plot)

        row5_1, row5_2, row5_3, row5_4, row5_5 = st.columns(
            (.1, 1, .1, 1, .1))
        with row5_2:
            st.subheader('Custumer score compared to other customers')
            st.pyplot(plot_histo_with_hline(loan_proba_list, server_pred[0]))

        with row5_4:

            feature_imp = pd.DataFrame(
                sorted(
                    zip(pipeline[3].feature_importances_, customer_X.columns)),
                columns=['Value', 'Feature'])

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(
                by="Value", ascending=False)[:10])
            plt.title('LightGBM Features (avg over folds)')
            st.pyplot(fig)

        row6_1, row6_2, row6_3, row6_4, row6_5 = st.columns(
            (.1, 1, .1, 1, .1))

        with row6_2:
            feature_list1 = shap_sorted_values_df.index[0:10]
            feature1 = st.selectbox(
                "Select first feature to see distribution",
                feature_list1)

        with row6_4:
            feature_list2 = shap_sorted_values_df.index[0:10]
            feature2 = st.selectbox(
                "Select second feature to see distribution",
                feature_list2, index=1)

        # Add visualization button
        visualization_btn = st.button('Visualize')
        # When visualization button is activate
        if visualization_btn or ('visualize' in st.session_state and
                                 st.session_state['visualize'] == idx_curr):
            st.session_state['visualize'] = idx_curr
            row7_1, row7_2, row7_3, row7_4, row7_5 = st.columns(
                (.1, 1, .1, 1, .1))
            with row7_2:
                st.subheader('Customer position for '+feature1)
                st.pyplot(plot_histo_with_hline(
                    X_df[feature1].dropna(), X_df.loc[idx_curr, feature1]))

            with row7_4:
                st.subheader('Customer position for '+feature2)
                st.pyplot(plot_histo_with_hline(
                    X_df[feature2].dropna(), X_df.loc[idx_curr, feature2]))

            row8_1, row8_2, row8_3 = st.columns((1, 1, 1))
            with row8_2:
                st.subheader(feature2+' VS ' + feature2+' with customer position')
                fig, ax = plt.subplots(figsize=(10, 10))
                sns.scatterplot(
                    x=X_df[feature1], y=X_df[feature2], hue=loan_proba_list)
                ax.scatter(X_df.loc[idx_curr, feature1],
                           X_df.loc[idx_curr, feature2], s=100, color="red")
                st.pyplot(fig)
