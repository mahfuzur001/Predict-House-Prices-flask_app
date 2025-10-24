import pandas as pd
import os, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ✅ CSV লোড
df = pd.read_csv(r'F:\daffodil\dipti\flask\Housing.csv')

# ✅ Scale numeric columns
scaler = MinMaxScaler()
df[['price', 'area']] = scaler.fit_transform(df[['price', 'area']])

# ✅ Encode categorical columns
categorical_cols = ['mainroad','guestroom','basement','hotwaterheating',
                    'airconditioning','prefarea','furnishingstatus']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ✅ Split features and target
x = df.drop('price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# ✅ Train model
lr = LinearRegression()
lr.fit(x_train, y_train)

# ✅ Save model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
pickle.dump(lr, open(model_path, 'wb'))

print("✅ Model saved as model.pkl")

# ✅ Load model again
loaded_model = pickle.load(open(model_path, 'rb'))
print("✅ Model loaded successfully!")

# ⚠️ Test prediction (example with all features)
sample = [list(x_test.iloc[0])]   # one row from test data
test_pred = loaded_model.predict(sample)
print("💰 Predicted Price:", round(test_pred[0], 2))
print("💰 Actual Price:", round(y_test.iloc[0], 2))