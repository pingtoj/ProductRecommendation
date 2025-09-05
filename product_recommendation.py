import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Load dataset
df = pd.read_csv("/content/ProductRecommendation.csv")
df

# Encode categorical variables
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

le_category = LabelEncoder()
df["ProductCategory"] = le_category.fit_transform(df["ProductCategory"])

le_purchased = LabelEncoder()
df["Purchased"] = le_purchased.fit_transform(df["Purchased"])  # Yes=1, No=0

# Features and target
X = df[["Age", "Gender", "Income", "ProductCategory"]]
y = df["Purchased"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred,zero_division=0))

# Show decision rules
rules = export_text(clf, feature_names=list(X.columns))
print(rules)

# Example prediction
sample = [[30, le_gender.transform(["Male"])[0], 35000, le_category.transform(["Electronics"])[0]]]

sample_df = pd.DataFrame(sample, columns=["Age", "Gender", "Income", "ProductCategory"])
prediction = clf.predict(sample_df)
print("Purchase Prediction:", le_purchased.inverse_transform(prediction)[0])
















