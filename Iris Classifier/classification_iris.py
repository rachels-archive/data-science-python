#import dependencies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** species!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Select Sepal Length', 0.0, 10.0)
    sepal_width = st.sidebar.slider('Select Sepal Width', 0.0, 10.0)
    petal_length = st.sidebar.slider('Select Petal Length', 0.0, 10.0)
    petal_width = st.sidebar.slider('Select Petal Width', 0.0, 10.0)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

#load data set
data = sns.load_dataset("iris")
data.head()

#prepare training set
# X = feature set
X = data.iloc[:, :-1] #Every column except last column

# y = target value
y = data.iloc[:, -1] # All rows for last column

#split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
model = LogisticRegression()
model.fit(x_train, y_train)

#test the model
predictions = model.predict(x_test)
prediction = model.predict(df)

if st.button("Predict"):
    st.success(prediction)

st.subheader('Confusion Metrics')
report = classification_report(y_test, predictions, output_dict=True)
report_df = df = pd.DataFrame(report).transpose()
st.write(report_df)
