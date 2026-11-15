import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop the leftover index column and the customer identifier
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# merge `Fe Male` typo with Female column
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Train shape:", Xtrain.shape, "| Test shape:", Xtest.shape)
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
