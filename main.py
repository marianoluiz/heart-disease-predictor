import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model('heartrisk_detector_model.keras')
# model.summary()

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
print(x)

# Define the data
data = [
    # Underweight young adult female with high mental distress, doesn’t smoke or drink, average sleep. Generally fair health.
    [17.8, 2.0, 25.0, 4, 3, 6.0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    # Healthy Asian male with no reported health issues, sleeps well, no smoking/drinking, excellent general health.
    [22.5, 0.0, 0.0, 8, 4, 9.0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    # Obese Black male smoker with stroke history and mobility issues, poor general health and low sleep.
    [35.6, 12.0, 10.0, 9, 1, 4.0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    # Normal BMI Hispanic female, no health issues, doesn’t smoke or drink, high general health.
    [26.2, 0.0, 0.0, 12, 2, 7.0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    # Obese male drinker, severe physical/mental distress, poor general health, limited mobility.
    [30.1, 30.0, 20.0, 11, 0, 5.0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    # Underweight Hispanic female, light physical/mental health issues, sleeps well, nonsmoker.
    [19.4, 1.0, 2.0, 5, 3, 10.0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    # Severely obese Black male smoker, poor physical and mental health, limited mobility, poor general health.
    [41.2, 6.0, 15.0, 10, 1, 6.0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    # Overweight female from Other race group, no issues, good sleep and health, nonsmoker.
    [28.0, 0.0, 0.0, 6, 2, 8.0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    # Normal weight Native American female, mild physical issues, fair general health, sleeps okay.
    [24.3, 3.0, 0.0, 7, 3, 7.0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    # Obese Hispanic male drinker, major health issues and stroke, limited mobility, poor health.
    [38.7, 20.0, 30.0, 9, 0, 4.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
]

# Define the column names
columns = [
    "bmi", "physical_health", "mental_health", "age_category", "gen_health", "sleep_time",
    "smoking_No", "smoking_Yes", "alcohol_drinking_No", "alcohol_drinking_Yes",
    "stroke_No", "stroke_Yes", "diff_walking_No", "diff_walking_Yes",
    "sex_Female", "sex_Male",
    "race_American Indian/Alaskan Native", "race_Asian", "race_Black", "race_Hispanic",
    "race_Other", "race_White", "diabetic_No", "diabetic_No, borderline diabetes", "diabetic_Yes",
    "diabetic_Yes (during pregnancy)", "physical_activity_No", "physical_activity_Yes",
    "asthma_No", "asthma_Yes", "kidney_disease_No", "kidney_disease_Yes",
    "skin_cancer_No", "skin_cancer_Yes"
]

# Create the DataFrame
x = pd.DataFrame(data, columns=columns)

prediction = model.predict(x)
print(prediction)