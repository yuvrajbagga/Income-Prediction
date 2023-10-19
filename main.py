import pickle
import sklearn
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load your pre-trained model
pree = pickle.load(open('classifier.pkl','rb'))

# Create a Streamlit app
st.title("Income Prediction")

# Add widgets to take user inputs if applicable
user_input = st.number_input("Age:",value=0, step=1)
user_input2 = st.selectbox("Workclass:", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
user_input3 = st.number_input("Final weight	:",value=0, step=1)
user_input4 = st.selectbox("Education:", ['Doctorate', '12th', 'Bachelors', '7th-8th', 'Some-college',
       'HS-grad', '9th', '10th', '11th', 'Masters', 'Preschool',
       '5th-6th', 'Prof-school', 'Assoc-voc', '1st-4th', 'Assoc-acdm'])
user_input5 = st.number_input("Educational-num:",value=0, step=1)
user_input6 = st.selectbox("Marital-Status:", ['Divorced', 'Never-married', 'Married-civ-spouse', 'Widowed',
       'Separated', 'Married-spouse-absent', 'Married-AF-spouse'])
user_input7 = st.selectbox("Occupation:", ['Exec-managerial', 'Other-service', 'Transport-moving',
       'Adm-clerical', 'Machine-op-inspct', 'Sales', 'Handlers-cleaners',
       'Farming-fishing', 'Protective-serv', 'Prof-specialty',
       'Craft-repair', np.nan, 'Tech-support', 'Priv-house-serv',
       'Armed-Forces'])
user_input8 = st.selectbox("Relationship:", ['Not-in-family', 'Own-child', 'Husband', 'Wife', 'Unmarried',
       'Other-relative'])
user_input9 = st.selectbox("Race:", ['White', 'Black', 'Asian-Pac-Islander', 'Other',
       'Amer-Indian-Eskimo'])
user_input10 =st.selectbox("Gender:", ['Male','Female'])
user_input11 = st.number_input("Capital-gain:",value=0, step=1)
user_input12 = st.number_input("Capital-loss:",value=0, step=1)
user_input13 = st.number_input("Hours-per-week:",value=0, step=1)
user_input14= st.selectbox("Native-Country:", ['United-States', 'Japan', 'South', 'Portugal', 'Italy', 'Mexico',
       'Ecuador', 'England', 'Philippines', 'China', 'Germany', np.nan,
       'Dominican-Republic', 'Jamaica', 'Vietnam', 'Thailand',
       'Puerto-Rico', 'Cuba', 'India', 'Cambodia', 'Yugoslavia', 'Iran',
       'El-Salvador', 'Poland', 'Greece', 'Ireland', 'Canada',
       'Guatemala', 'Scotland', 'Columbia', 'Outlying-US(Guam-USVI-etc)',
       'Haiti', 'Peru', 'Nicaragua', 'Taiwan', 'France',
       'Trinadad&Tobago', 'Laos', 'Hungary', 'Honduras', 'Hong',
       'Holand-Netherlands'])



Workclass=['Private', 'State-gov', 'Self-emp-not-inc', 'Federal-gov',
       'Local-gov', 'Self-emp-inc', np.nan, 'Never-worked', 'Without-pay']

Gender=['Male', 'Female']
Native_country=['United-States', 'Japan', 'South', 'Portugal', 'Italy', 'Mexico',
       'Ecuador', 'England', 'Philippines', 'China', 'Germany', np.nan,
       'Dominican-Republic', 'Jamaica', 'Vietnam', 'Thailand',
       'Puerto-Rico', 'Cuba', 'India', 'Cambodia', 'Yugoslavia', 'Iran',
       'El-Salvador', 'Poland', 'Greece', 'Ireland', 'Canada',
       'Guatemala', 'Scotland', 'Columbia', 'Outlying-US(Guam-USVI-etc)',
       'Haiti', 'Peru', 'Nicaragua', 'Taiwan', 'France',
       'Trinadad&Tobago', 'Laos', 'Hungary', 'Honduras', 'Hong',
       'Holand-Netherlands']



Occupation=['Exec-managerial', 'Other-service', 'Transport-moving',
       'Adm-clerical', 'Machine-op-inspct', 'Sales', 'Handlers-cleaners',
       'Farming-fishing', 'Protective-serv', 'Prof-specialty',
       'Craft-repair', np.nan, 'Tech-support', 'Priv-house-serv',
       'Armed-Forces']

Education=['Doctorate', '12th', 'Bachelors', '7th-8th', 'Some-college',
       'HS-grad', '9th', '10th', '11th', 'Masters', 'Preschool',
       '5th-6th', 'Prof-school', 'Assoc-voc', '1st-4th', 'Assoc-acdm']

relationship=['Not-in-family', 'Own-child', 'Husband', 'Wife', 'Unmarried',
       'Other-relative']

marital_status=['Divorced', 'Never-married', 'Married-civ-spouse', 'Widowed',
       'Separated', 'Married-spouse-absent', 'Married-AF-spouse']
Race=['White', 'Black', 'Asian-Pac-Islander', 'Other',
       'Amer-Indian-Eskimo']

def enc(name,user_input):
    label_enc = LabelEncoder()
    label_enc.fit(name)
    name = label_enc.transform(name)
    user_input=label_enc.transform([user_input])
    return name , user_input[0]


user_input2=enc(Workclass,user_input2)[1]
user_input10=enc(Gender,user_input10)[1]
user_input14=enc(Native_country,user_input14)[1]
user_input7=enc(Occupation,user_input7)[1]
user_input4=enc(Education,user_input4)[1]
user_input8=enc(relationship,user_input8)[1]
user_input6=enc(marital_status,user_input6)[1]
user_input9=enc(Race,user_input9)[1]


data=[user_input,user_input2,user_input3,user_input4,user_input5,user_input6,user_input7,user_input8,user_input9,user_input10,user_input11,user_input12,user_input13,user_input14]
# Make pr,edictions using your model
if st.button("Predict"):
    prediction = pree.predict([data])
    if prediction[0]==0:
        st.write('Prediction: Income is less then 50k USD')
    else:
        st.write('Prediction: Income is more then 50k USD')
        

# Add any other Streamlit components to visualize result