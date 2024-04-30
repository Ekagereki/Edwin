import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("mother_survey.csv")

# replacing -99, 9999 & -88 with NA
df.replace({9999: pd.NA, -99: pd.NA, -88: pd.NA, 'NA': pd.NA}, inplace=True)
df.to_csv("cleaned_data.csv", index=False)

# loading cleaned dataset
df1 = pd.read_csv("cleaned_data.csv")

# descriptive statistics
summary_table = df1.describe()
# print(summary_table)

# boxplot of the percentage of children enrolled in school by gender.
data = pd.concat([df["n_children"], df.iloc[:, 13:25]], axis=1)
# using one code name for gender
data["sex_1"] = data["sex_1"].replace({'female': 'F', 'Female': 'F', 'F': 'F', 'male': 'M', 'Male': 'M', 'M': 'M'})
data["sex_2"] = data["sex_2"].replace({'female': 'F', 'Female': 'F', 'F': 'F', 'male': 'M', 'Male': 'M', 'M': 'M'})
data["sex_3"] = data["sex_3"].replace({'female': 'F', 'Female': 'F', 'F': 'F', 'male': 'M', 'Male': 'M', 'M': 'M'})
data["sex_4"] = data["sex_4"].replace({'female': 'F', 'Female': 'F', 'F': 'F', 'male': 'M', 'Male': 'M', 'M': 'M'})
data["sex_5"] = data["sex_5"].replace({'female': 'F', 'Female': 'F', 'F': 'F', 'male': 'M', 'Male': 'M', 'M': 'M'})
data["sex_6"] = data["sex_6"].replace({'female': 'F', 'Female': 'F', 'F': 'F', 'male': 'M', 'Male': 'M', 'M': 'M'})

data.replace({'NA': pd.NA}, inplace=True)

for i in range(1, 7):
    data[f'child_{i}_enrolled_in_school'] = data.apply(lambda x: x[f'sex_{i}'] if x[f'in_school_{i}'] == 1 else None,
                                                       axis=1)

data.to_csv("box_plot_data.csv", index=False)
# selecting columns N:S from the box_plot_data csv
data = pd.read_csv("box_plot_data.csv")
df2 = data.iloc[:, 13:18]

# Extracting genders from column names
genders = [col.split('_')[1] for col in df2.columns]

# Plotting box plot
plt.figure(figsize=(10, 10))
plt.boxplot([df2[col].value_counts().dropna() for col in df2.columns], labels=genders)
plt.title('Enrolled in School by Gender')
plt.xlabel('enrollment per order of child born')
plt.ylabel('Count')
plt.show()

# regression estimation y = mx + c
# obtaining a subset of relevant data
data2 = pd.concat([df1.iloc[:, 3:7], df1.iloc[:, 19:25]], axis=1)
df3 = data2.iloc[:, 5:10]

# Combine columns into one categorical variable
df3['enrolled'] = df3.apply(lambda row: 'yes' if 1 in row.values else 'no', axis=1)

# Calculate counts
enrollment_counts = df3['enrolled'].value_counts()

# Combine enrollment data with independent variables
combined_df = pd.concat([df3['enrolled'], data2], axis=1)
combined_df = combined_df.fillna(0)
# Prepare data for logistic regression
X = combined_df.drop(['enrolled', 'in_school_1', 'in_school_2', 'in_school_3', 'in_school_4', 'in_school_5',
                      'in_school_6'], axis=1)
y = combined_df['enrolled']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Get the coefficients and intercept from the trained model
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Create a DataFrame to store the coefficients
coefficients_df = pd.DataFrame({'Feature': X.columns.tolist(), 'Coefficient': coefficients})

# Add the intercept as a row in the DataFrame
# Add the intercept to the DataFrame
intercept_df = pd.DataFrame({'Feature': ['Intercept'], 'Coefficient': [model.intercept_[0]]})
coefficients_df = pd.concat([intercept_df, coefficients_df])

# Export the DataFrame to a table (CSV file in this case)
coefficients_df.to_csv('logistic_regression_results.csv', index=False)

# Display the DataFrame
print(coefficients_df)
