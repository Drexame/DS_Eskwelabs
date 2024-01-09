import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors

# Loading .joblib and funded loans file
model = joblib.load('kiva_model_xgb.joblib')
recommender = joblib.load('kiva_recommender_2.joblib')
funded_loans_only = pd.read_csv('funded_loans_ph_original.csv')

# Text variables
app_description = 'Boost your Kiva loan success with Ka-a-Kiva-t – the ultimate companion app. Unlock funding potential through personalized tips and analytics, ensuring your loans receive the attention they deserve.'
similar_funded_loans_description = 'Here are 5 similar loans that were funded given the loan characteristics input:'
summary_description = 'Here is the summary of the 5 similar loans:'

# Functions used
def repayment_term_16_mos_and_above_value(repayment_term_16_mos_and_above_choice):
    if repayment_term_16_mos_and_above_choice == 'Yes':
        repayment_term_16_mos_and_above = 1
    else:
        repayment_term_16_mos_and_above = 0

    return repayment_term_16_mos_and_above

def partner_covers_currency_loss_value(partner_covers_currency_loss_choice):
    if partner_covers_currency_loss_choice == 'Yes':
        partner_covers_currency_loss = 1
    else:
        partner_covers_currency_loss = 0

    return partner_covers_currency_loss

def with_image_value(with_image_choice):
    if with_image_choice == 'Yes':
        with_image = 1
    else:
        with_image = 0

    return with_image

def with_video_value(with_video_choice):
    if with_video_choice == 'Yes':
        with_video = 1
    else:
        with_video = 0

    return with_video

def repayment_interval_value(repayment_interval_choice):
    if repayment_interval_choice == 'Irregular':
        repayment_interval_irregular = 1
        repayment_interval_monthly = 0
    else:
        repayment_interval_irregular = 0
        repayment_interval_monthly = 1

    return repayment_interval_irregular, repayment_interval_monthly

def sector_name_values(sector_name_choice):
    if sector_name_choice == 'Arts':
        sector_name_Arts = 1
        sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Construction':
        sector_name_Construction = 1
        sector_name_Arts, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Education':
        sector_name_Education = 1
        sector_name_Arts, sector_name_Construction, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Health':
        sector_name_Health = 1
        sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Housing':
        sector_name_Housing = 1
        sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Manufacturing':
        sector_name_Manufacturing = 1
        sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Personal Use':
        sector_name_Personal_Use = 1
        sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Retail, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Retail':
        sector_name_Retail = 1
        sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Services = 0, 0, 0, 0, 0, 0, 0, 0
    elif sector_name_choice == 'Services':
        sector_name_Services = 1
        sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail = 0, 0, 0, 0, 0, 0, 0, 0

    return sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services

def prediction_value(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p):

    prediction = model.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]])
    
    if prediction[0] == 1:
        prediction_text = 'Result: The loan is funded'
    else:
        prediction_text = 'Result: The loan is not funded'
    
    return prediction_text

def recommended_loans(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p):
    user_input = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p]
    user_input_array = np.array(user_input)

    _, recommendations = recommender.kneighbors(user_input_array.reshape(1, -1), n_neighbors=5)

    reco = recommendations[0].tolist()
    funded_reco = pd.DataFrame()

    for x in reco:
        funded_reco = funded_reco.append(funded_loans_only.loc[x], ignore_index=True)

    return funded_reco

def on_button_click(prediction_text, similar_funded_loans):
    # Centered and bigger text using Markdown and HTML
    centered_and_bigger_text = f"<h1 style='text-align: center; color: black;'>{prediction_text}</h1>"

    # Display the centered and bigger text
    st.markdown(centered_and_bigger_text, unsafe_allow_html=True)

    # Display 5 funded loans that have similar characteristics to the user's input
    st.markdown(f"<p style='font-size: 22px;'>{similar_funded_loans_description}</p>", unsafe_allow_html=True)
    st.dataframe(similar_funded_loans)

    # Manual text summary
    loan_amount_avg = similar_funded_loans['loan_amount'].mean()
    num_days_to_fund_avg = similar_funded_loans['num_days_to_fully_fund'].mean()
    sector_mode = similar_funded_loans['sector_name'].value_counts().idxmax()
    repayment_interval_mode = similar_funded_loans['repayment_interval'].value_counts().idxmax()
    with_image_mode = similar_funded_loans['with_image'].value_counts().idxmax()

    if with_image_mode == 1:
        with_image_text = 'Most of the loans have an image attached to them.'
    else:
        with_image_text = 'Most of the loans do not have an image attached to them.'

    st.markdown(f"<p style='font-size: 22px;'>{summary_description}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px;'>1. The average loan amount for the loans are {loan_amount_avg}.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px;'>2. The average number of days to fund the loans are {num_days_to_fund_avg}.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px;'>3. The most common sector among the loans is '{sector_mode}'.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px;'>4. The most common repayment interval is {repayment_interval_mode}.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px;'>5. {with_image_text}</p>", unsafe_allow_html=True)


def main():
    # Streamlit app title
    st.markdown("<h1 style='color: #2aa967; text-align: center;'>Ka-a-KIVA-t</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: black; text-align: justify; font-size: 24px;'>{app_description}</h3>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

        # Variable order in model
            # "loan_amount",
            # "repayment_term_16_mos_and_above",
            # "partner_covers_currency_loss",
            # "with_image",
            # "with_video",
            # "repayment_interval_irregular",
            # "repayment_interval_monthly",
            # 'sector_name_Arts', 'sector_name_Construction',
            # 'sector_name_Education',
            # 'sector_name_Health', 'sector_name_Housing', 'sector_name_Manufacturing',
            # 'sector_name_Personal_Use', 'sector_name_Retail', 'sector_name_Services'

    # User Inputs
    col1, col2 = st.columns(2)

    with col1:
        st.header("Numerical Inputs:")
        loan_amount = st.number_input(label = 'Enter loan amount:')

    with col2:
        st.header("Categorical Inputs:")
        repayment_term_16_mos_and_above_choice = st.selectbox('Is the repayment 16 months and above?:', ['Yes', 'No'])
        partner_covers_currency_loss_choice = st.selectbox('Does partner cover currency loss?:', ['Yes', 'No'])
        with_image_choice = st.selectbox('Do they have an image uploaded with the loan?:', ['Yes', 'No'])
        with_video_choice = st.selectbox('Do they have a video uploaded with the loan?:', ['Yes', 'No'])
        repayment_interval_choice = st.selectbox('What is the repayment interval?:', ['Irregular', 'Monthly'])
        sector_name_choice = st.selectbox('What sector is the loan a part of?:', ['Arts', 'Construction', 'Education', 'Health', 'Housing', 'Manufacturing', 'Personal Use', 'Retail', 'Services'])

    # Giving categorical variables a value based on user's input
    repayment_term_16_mos_and_above = repayment_term_16_mos_and_above_value(repayment_term_16_mos_and_above_choice)
    partner_covers_currency_loss = partner_covers_currency_loss_value(partner_covers_currency_loss_choice)
    with_image = with_image_value(with_image_choice)
    with_video = with_video_value(with_video_choice)
    repayment_interval_irregular, repayment_interval_monthly = repayment_interval_value(repayment_interval_choice)
    sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services = sector_name_values(sector_name_choice)

    # Getting prediction
    prediction_text = prediction_value(loan_amount, repayment_term_16_mos_and_above, partner_covers_currency_loss, with_image, with_video, repayment_interval_irregular, repayment_interval_monthly, sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services)
    similar_funded_loans = recommended_loans(loan_amount, repayment_term_16_mos_and_above, partner_covers_currency_loss, with_image, with_video, repayment_interval_irregular, repayment_interval_monthly, sector_name_Arts, sector_name_Construction, sector_name_Education, sector_name_Health, sector_name_Housing, sector_name_Manufacturing, sector_name_Personal_Use, sector_name_Retail, sector_name_Services)

    # Button to check whether loan is funded or not
    col3, col4, col5 = st.columns([1,2,1])
    
    with col3:
        pass
    with col4:
        predict_button = st.button("Click me to see if loan is funded or not")
    with col5:
        pass
    
    if predict_button:
        on_button_click(prediction_text, similar_funded_loans)


if __name__ == "__main__":
    main()
