import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("❌ Error: dataset.csv file not found.")
    exit()

# Convert categorical columns to numeric
df['Interest'] = df['Interest'].astype('category').cat.codes
df['Career_Path'] = df['Career_Path'].astype('category')
target_names = df['Career_Path'].cat.categories
df['Career_Path'] = df['Career_Path'].cat.codes

# Define features and target
X = df[['Age', 'Marks_Math', 'Marks_Eng', 'Interest']]
y = df['Career_Path']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and labels
with open('career_model.pkl', 'wb') as f:
    pickle.dump((model, target_names), f)

print("✅ Model trained and saved as career_model.pkl successfully.")

