# Logistic-regression
#regression with python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def create_dataset():
    # Creating the dataset
    data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06'],
        'Product': ['Widget', 'Widget', 'Gadget', 'Widget', 'Gadget', 'Widget'],
        'Amount': [100, 150, 200, 120, 180, 130],
        'Quantity': [2, 3, 5, 4, 2, 3]
    }

    # Convert 'Date' to datetime format and then to numerical (timestamp)
    data['Date'] = pd.to_datetime(data['Date'])

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # One-Hot Encoding for 'Product' column
    df = pd.get_dummies(df, columns=['Product'], drop_first=True)

    # Extract the 'Date' column in numeric form (timestamp)
    df['Date'] = df['Date'].astype(int) / 10**9  # Convert to seconds since epoch

    print(df.describe())
    print(df.info())

    # Defining features (X) and target (Y)
    X = df[['Date', 'Amount', 'Product_Widget']]  # 'Product_Widget' is the encoded column for 'Widget'
    Y = df['Quantity']

    print(X)
    print(Y)

    # Scatter plot of the dataset
    sns.scatterplot(x='Amount', y='Quantity', data=df)
    plt.show()

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Linear Regression Model
    print("Linear Regression:")
    regr = LinearRegression()
    regr.fit(X_train, Y_train)
    print(f"Linear Regression R^2 Score: {regr.score(X_test, Y_test)}")

    # K-Nearest Neighbors Regression Model
    print("\nK-Nearest Neighbors Regression:")
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, Y_train)
    print(f"KNN Regression R^2 Score: {knn.score(X_test, Y_test)}")

    # Logistic Regression Model (using classification for simplicity, thresholding quantity into classes)
    print("\nLogistic Regression (classification):")
    # Converting 'Quantity' to a binary classification problem for Logistic Regression
    Y_class = (Y > 3).astype(int)  # For example, classify quantities > 3 as 1 (class), others as 0
    X_train, X_test, Y_class_train, Y_class_test = train_test_split(X, Y_class, test_size=0.2, random_state=42)

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_class_train)
    y_pred = logreg.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(Y_class_test, y_pred)}")

# Call the function to create dataset and perform analysis
create_dataset()

