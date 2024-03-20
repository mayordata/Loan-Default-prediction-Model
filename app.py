import streamlit as st
import pandas as pd
import numpy as np
#import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder



# Load the model

model = load_model('loanmodel4.keras')

# model = joblib.load('loanmodel3.joblib')

# creating preprocessing function
def preprocess_data(Age,Income,LoanAmount,CreditScore, 
                    MonthsEmployed,NumCreditLines,InterestRate, 
                    LoanTerm, DTIRatio,Education,EmploymentType, 
                    MaritalStatus,HasMortgage,HasDependents,LoanPurpose,HasCoSigner):
    
    # create dataframe with the user input
    data = pd.DataFrame({
        'Age' : [Age],
        'Income': [Income],
        'LoanAmount': [LoanAmount],
        'CreditScore': [CreditScore],
        'MonthsEmployed': [MonthsEmployed],
        'NumCreditLines': [NumCreditLines],
        'InterestRate': [InterestRate],
        'LoanTerm': [LoanTerm],
        'DTIRatio': [DTIRatio],
        'Education': [Education],
        'EmploymentType': [EmploymentType],
        'MaritalStatus': [MaritalStatus],
        'HasMortgage': [HasMortgage],
        'HasDependents': [HasDependents],
        'LoanPurpose': [LoanPurpose],
        'HasCoSigner': [HasCoSigner] 
    })

    # Encode binary columns
    data['HasMortgage'] =  data['HasMortgage'].map({'Yes': 1, 'No': 0})
    data['HasDependents'] = data['HasDependents'].map({'Yes': 1, 'No': 0})
    data['HasCoSigner'] =  data['HasCoSigner'].map({'Yes': 1, 'No': 0})

    # mapping my categorical columns
    
    data['Education'] = data['Education'].map({"Bachelor\'s": 0, 'High School': 1, "Master\'s": 2, 'PhD': 3})
    data['EmploymentType'] = data['EmploymentType'].map({'Full-time': 0, 'Part-time': 1, 'Self-employed': 2, 'Unemployed': 3})
    data['MaritalStatus'] = data['MaritalStatus'].map({'Divorced': 0, 'Married': 1, 'Single': 2})
    data['LoanPurpose'] = data['LoanPurpose'].map({'Auto': 0, 'Business': 1, 'Education': 2, 'Home':3, 'Other' :4})


    # Encoding Categorical columns

    #Le = LabelEncoder()

    #data['Education'] = Le.fit_transform( data['Education'])
    #data['EmploymentType'] = Le.fit_transform( data['EmploymentType'])
    #data['MaritalStatus'] = Le.fit_transform( data['MaritalStatus'])
    #data['LoanPurpose'] = Le.fit_transform( data['LoanPurpose'])
  
    # scale my numerical model 
    scaler = MinMaxScaler()
    numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data

# Creating the main function
def main():
    st.title('Loan Prediction App')
    st.write('This is a loan default prediction app. please enter the requied information to get the prediction')
    st.image('download.jpeg', use_column_width = True)
    st.subheader('Please enter the required information')
    st.divider()

    # User prompt
    Age = st.slider('Age', min_value = 18, max_value = 100, value = 25)
    Income = st.number_input('Income', value = 30000)
    LoanAmount = st.number_input('Loan Amount', value = 30000)
    CreditScore = st.slider('Credi tScore', min_value = 300, max_value = 850, value = 450)
    MonthsEmployed = st.number_input('Months Employed', value = 12)
    NumCreditLines = st.slider('Number of CreditLines', min_value = 1, max_value = 4, value = 2)
    InterestRate = st.slider('InterestRate', min_value = 0, max_value = 30, value = 15)
    LoanTerm = st.slider('LoanTerm', min_value = 12, max_value = 60, value = 24)
    DTIRatio = st.slider('DTIRatio', min_value = 0.0, max_value = 1.0, value = 0.5)

    Education = st.selectbox('Education', ["Bachelor\'s", "High School", "Master\'s", "PhD"])
    EmploymentType = st.selectbox('Employment Type', ["Unemployed", "employed", "Self-employed", "Full-time"])
    MaritalStatus = st.selectbox('Marital Status', ["Married", "Divorced", "Single"])
    HasMortgage = st.radio('HasMortatge', ["Yes", "No"])
    HasDependents = st.radio('Has Dependent', ['Yes', 'No'])
    LoanPurpose = st.selectbox('Loan Purpose', ['Business', 'Home', 'Education', 'Other', 'Auto'])
    HasCoSigner = st.radio('Has Cosigner', ['Yes', 'No'])

    # Preprocess the user input 
    #user_data = preprocess_data(Age,Income,LoanAmount,CreditScore, 
                    #MonthsEmployed,NumCreditLines, InterestRate, 
                    #LoanTerm,DTIRatio,Education,EmploymentType, 
                   # MaritalStatus,HasMortgage,HasDependents,LoanPurpose,HasCoSigner)
    
    # Make Prediction with loaded model
    #prediction = model.predict(user_data)
    

    # Display the prediction
   
    #if prediction > 0.5:
    #    st.write('The loan will default')
    #else:
    #    st.write('The loan will not default')

    if st.button('Predict'):
        user_data = preprocess_data(Age,Income,LoanAmount,CreditScore, 
                    MonthsEmployed,NumCreditLines, InterestRate, 
                    LoanTerm,DTIRatio,Education,EmploymentType, 
                    MaritalStatus,HasMortgage,HasDependents,LoanPurpose,HasCoSigner)
        
        prediction = model.predict(user_data)

        if prediction > 0.5:
           st.success('The loan will default')
        else:
          st.error('The loan will not default')













if __name__ == '__main__':
    main()
