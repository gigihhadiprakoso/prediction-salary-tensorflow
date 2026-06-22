import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv('assets/DataScience_salaries_2024.csv')

# Preprocessing
# a. Handle categorical features
categorical_features = ['experience_level', 'employment_type', 'job_title', 'company_location', 'company_size']
numeric_features = ['work_year']

# b. Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# c. Bagi data
X = df.drop('salary_in_usd', axis=1)
y = df['salary_in_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# d. Fitting preprocessor
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_preprocessed.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train_preprocessed,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)


test_loss, test_mae = model.evaluate(X_test_preprocessed, y_test)
print(f"Test MAE: {test_mae}")

def predict_salary(input_data):
    """
    Contoh input:
    {
        'work_year': 2023,
        'experience_level': 'SE',
        'employment_type': 'FT',
        'job_title': 'Data Scientist',
        'company_location': 'US',
        'company_size': 'L'
    }
    """
    # Convert input ke DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocessing
    processed_input = preprocessor.transform(input_df)
    
    # Prediksi
    prediction = model.predict(processed_input)
    
    return prediction[0][0]


sample_input = {
    'work_year': 2024,
    'experience_level': 'SE',
    'employment_type': 'FT',
    'job_title': 'Data Scientist',
    'company_location': 'CH',
    'company_size': 'L'
}

predicted_salary = predict_salary(sample_input)
print(f"Predicted Salary: ${predicted_salary:,.2f}")