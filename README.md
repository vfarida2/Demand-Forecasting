# Demand-Forecasting
#End-to-end demand forecasting project using ML models (Linear Regression, Random Forest, XGBoost) with data analysis and visualization.

#Data loading & inspection
df=pd.read_csv('/content/clean_data.csv')
df.info()
df.describe()
df.isnull().sum()

#Remove duplicates
df.drop_duplicates(inplace=True)

#Duplicate rows are removed to ensure data quality.
#Handle missing values
df["region"].fillna(df["region"].mode()[0], inplace=True)
df["price"].fillna(df["price"].mean(), inplace=True)
#Categorical values → filled with mode
#Numerical values → filled with mean

#Replace incorrect zero values
df["region"].replace(0, df["region"].mode()[0], inplace=True)
df["price"].replace(0, df["price"].median(), inplace=True)

#Zero values that are considered invalid are replaced appropriately.
#Drop unnecessary column
df.drop('Epidemic', axis=1, inplace=True)

#Convert and extract date features
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

#The date column is transformed into useful features for modeling:

Year
Month
Day

#Outlier Detection (IQR Method)
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

#Outliers are identified using the IQR method.

outliers = df[(df["price"] < lower) | (df["price"] > upper)]

📌 Note:

#Outliers were not removed as they represent real-world variations in demand.

#Encoding
df = pd.get_dummies(df, drop_first=True)

#Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)
#80% training / 20% testing
#shuffle=False is used to preserve time order (important for forecasting)

#Model Training & Evaluation
🔹 Linear Regression
MAE = 25.89

#A simple baseline model. Performance is relatively weak due to linear assumptions.

🔹 Random Forest
MAE = 19.17

#The best-performing model. It captures non-linear relationships effectively.

🔹 XGBoost
MAE = 20.48

#A powerful boosting model with strong performance, slightly below Random Forest.

#Model Comparison
results = pd.DataFrame({
    'Actual': y_test,
    'LR': y_pred_lr,
    'RF': y_pred_rf,
    'XGB': y_pred_xgb
})

#This table compares actual values with predictions from each model.

Categorical variables are converted into numerical format using one-hot encoding.

Key Insights
- Random Forest achieved the best performance with the lowest MAE.
- Linear Regression struggled to capture complex patterns in the data.
- XGBoost performed well but showed higher variance in some cases.
- All models tend to perform worse on very low or irregular demand values.
