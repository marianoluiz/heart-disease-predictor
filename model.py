import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# declare the dataset
dataset = pd.read_csv('./heartdisease2020_dataset_by_kamil/heart_2020_cleaned.csv')
dataset.rename(columns={
    'HeartDisease': 'heart_disease',
    'BMI': 'bmi',
    'Smoking': 'smoking',
    'AlcoholDrinking': 'alcohol_drinking',
    'Stroke': 'stroke',
    'PhysicalHealth': 'physical_health',
    'MentalHealth': 'mental_health',
    'DiffWalking': 'diff_walking',
    'Sex': 'sex',
    'AgeCategory': 'age_category',
    'Race': 'race',
    'Diabetic': 'diabetic',
    'PhysicalActivity': 'physical_activity',
    'GenHealth': 'gen_health',
    'SleepTime': 'sleep_time',
    'Asthma': 'asthma',
    'KidneyDisease': 'kidney_disease',
    'SkinCancer': 'skin_cancer'
}, inplace=True)

age_map = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80 or older':  12,      
}

gen_health_map = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}

dataset['age_category'].replace(age_map, inplace=True)
dataset['gen_health'].replace(gen_health_map, inplace=True)
dataset['heart_disease'].replace({'Yes': 1, 'No': 0}, inplace=True)

x = dataset.drop(['heart_disease'], axis=1)
y = dataset['heart_disease']

x = pd.get_dummies(x, dtype=int)

pd.set_option('display.max_columns', None)

# print('x data: ', x)
# print('y data: ', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# declare the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(34, )),    # Hidden Layer
    tf.keras.layers.Dense(32, activation='relu'),                       # Hidden Layer
    tf.keras.layers.Dense(1, activation='sigmoid')                      # Output for Binary Classification
])

# Configure the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(x, y, epochs=8, validation_data=(x_test, y_test), callbacks=[early_stop])

# Evaluate the model
loss, accuracy =  model.evaluate(x_test, y_test)
print('The accuracy of the model is :', accuracy)

# Save the model
model.save('heartrisk_detector_model.keras')