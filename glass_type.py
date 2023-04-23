# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pydeck as pdk
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"

    df = pd.read_csv(file_path, header = None)
  
    
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['Reuse', 'Recycle', 'Reduce', 'Water', 'Electricty', 'E-Cars', 'Bicycle', 'Paper', 'SolarPanels', 'Carbon Emmision']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['Carbon Emmision']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# S3.1: Create a function that accepts an ML model object say 'model' and the nine features as inputs 
# and returns the glass type.

#features_col = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'] 
@st.cache() 
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "Building Windows Float Processed"
    elif glass_type == 2:
        return "Building Windows Non - Float Processed"
    elif glass_type == 3:
        return "Vehicle Windows Float Processed"
    elif glass_type == 4:
        return "Vehicle Windows Non - Float Processed"
    elif glass_type == 5:
        return "Containers"
    elif glass_type == 6:
        return "Table Ware"
    else:
        return "Head Lamps"
        
# S4.1: Add title on the main page and in the sidebar.
st.title("Carbon Emmision Prediction Web App")        
st.sidebar.title("Carbon Emmision Prediction Web App")
        
# S1.1: Add a multiselect widget to allow the user to select multiple visualisation.
st.sidebar.subheader("Visuliazation Selector")
plot_list = st.sidebar.multiselect('Select the Plots',("Corr_heatmap","line_chart",'area_chart','boxplot')) 
if 'line_chart' in plot_list:
    st.subheader("Line Chart")  
    st.line_chart(glass_df) 

if 'area_chart' in plot_list:
    st.subheader("Area Chart")
    st.area_chart(glass_df)

st.set_option('deprecation.showPyplotGlobalUse', False)
if "Corr_heatmap" in plot_list:
    st.subheader('Correlation Heatmap')
    plt.figure(figsize=(10,5))
    sns.heatmap(glass_df.corr(),annot = True)
    st.pyplot()

#if 'Count_plot' in plot_list:
    #st.subheader("Count Plot")
    #plt.figure(figsize=(10,5))
    #sns.countplot(x = "GlassType",data = glass_df)
    #st.pyplot()

#if 'piechart' in plot_list:
    #count_glass = glass_df["GlassType"].value_counts()
    #st.subheader('Pie Chart')
    #plt.figure(figsize=(10,5))
    #plt.pie(count_glass,labels = count_glass.index,autopct = '%1.2f%%')
    #st.pyplot()      
if "boxplot" in plot_list:
    st.subheader("Box Plot")
    column = st.sidebar.selectbox("Select The Column for Box Plot",('Reuse', 'Recycle', 'Reduce', 'Water', 'Electricty', 'E-Cars', 'Bicycle', 'Paper', 'SolarPanels', 'Carbon Emmision'))
    sns.boxplot(glass_df[column])
    st.pyplot()
# S2.1: Add 9 slider widgets for accepting user input for 9 features.
st.sidebar.subheader("Select your Values")
ri = st.sidebar.slider('Reuse',float(glass_df['Reuse'].min()),float(glass_df['Reuse'].max()))
na = st.sidebar.slider('Recycle',float(glass_df['Recycle'].min()),float(glass_df['Recycle'].max()))
mg = st.sidebar.slider('Reduce',float(glass_df['Reduce'].min()),float(glass_df['Reduce'].max()))
al = st.sidebar.slider('Water',float(glass_df['Water'].min()),float(glass_df['Water'].max()))
si = st.sidebar.slider('Electricty',float(glass_df['Electricty'].min()),float(glass_df['Electricty'].max()))
k = st.sidebar.slider('E-Cars',float(glass_df['E-Cars'].min()),float(glass_df['E-Cars'].max()))
ca = st.sidebar.slider('Bicycle',float(glass_df['Bicycle'].min()),float(glass_df['Bicycle'].max()))
ba = st.sidebar.slider('Paper',float(glass_df['Paper'].min()),float(glass_df['Paper'].max()))
fe = st.sidebar.slider('Carbon Emmision',float(glass_df['Carbon Emmision'].min()),float(glass_df['Carbon Emmision'].max()))

# S3.1: Add a subheader and multiselect widget.
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox('Classifier',("Support Vector Machine","Random Forest Classifier"))

# S4.1: Implement SVM with hyperparameter tuning
from sklearn.metrics import plot_confusion_matrix
if classifier == "Support Vector Machine":
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st.sidebar.number_input("C",1,100,step = 1)
    kernel_input = st.sidebar.radio("kernel",('linear','poly','rbf'))
    gamma_input = st.sidebar.number_input("gamma",1,100,step = 1)
    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model = SVC(C = c_value,kernel = kernel_input,gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test,y_test)
        glass_type = prediction(svc_model,ri, na, mg, al, si, k, ca,ba,fe)
        st.write('Type Of Glass Predicted is',glass_type)
        st.write("Accuracy for CarbonEmmsion Saved",accuracy.round(2))
        plot_confusion_matrix(svc_model,X_test,y_test)
        st.pyplot()

# S5.1: ImplementRandom Forest Classifier with hyperparameter tuning.
if classifier == "Random Forest Classifier":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of Attempt",100,5000,step = 10)
    max_depth_input = st.sidebar.number_input("Max depth",1,100,step = 1)
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rfc_model = RandomForestClassifier(n_estimators = n_estimators_input,max_depth = max_depth_input,n_jobs = -1)
        rfc_model.fit(X_train,y_train)
        y_pred = rfc_model.predict(X_test)
        accuracy = rfc_model.score(X_test,y_test)
        glass_type = prediction(rfc_model,ri, na, mg, al, si, k, ca,ba,fe)
        st.write('Type Of Glass Predicted is',glass_type)
        st.write("Accuracy for CarbonEmmision Save",accuracy.round(2))
        plot_confusion_matrix(rfc_model,X_test,y_test)
        st.pyplot()
