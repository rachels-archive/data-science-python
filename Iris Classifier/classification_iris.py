# import dependencies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.write("""
# Iris Flower Prediction App
This app predicts the Iris flower species based on user inputs using the **logistic regression** algorithm.
""")

image_url = "![Iris Species](https://www.researchgate.net/publication/349634676/figure/fig2/AS:995453013336067@1614345901799/Three-classes-of-IRIS-dataset-for-classification-17.jpg)"
st.markdown(image_url)


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

# load data set
data = sns.load_dataset("iris")
data.head()

# prepare training set
# X = feature set
X = data.iloc[:, :-1] #Every column except last column

# y = target value
y = data.iloc[:, -1] # All rows for last column

# split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# test the model
prediction = model.predict(df)

if st.button("Click here to classify"):
    st.success(prediction)

