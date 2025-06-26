import streamlit as st
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import os
import time
import sqlite3
from datetime import datetime

# ---- Database Setup ----
def init_database():
    """Initialize SQLite database - creates empty table structure"""
    conn = sqlite3.connect('user_database.db')
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pan_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            annual_income INTEGER NOT NULL,
            cibil_score INTEGER NOT NULL,
            previous_loan_status TEXT NOT NULL,
            created_date TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def get_user_by_pan(pan_number):
    """Retrieve user details by PAN number"""
    conn = sqlite3.connect('user_database.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE pan_number = ?', (pan_number,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'pan_number': result[1],
            'name': result[2],
            'age': result[3],
            'annual_income': result[4],
            'cibil_score': result[5],
            'previous_loan_status': result[6],
            'created_date': result[7]
        }
    return None

def check_database_status():
    """Check if database has data"""
    try:
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return 0

# Initialize database on startup
init_database()

# Check if database has data
customer_count = check_database_status()

# ---- Add calculate_emi function here, at the top ----
def calculate_emi(principal, annual_rate, tenure_years):
    """
    Calculate EMI, total interest, and total payment.

    principal: loan amount
    annual_rate: annual interest rate in percentage
    tenure_years: loan tenure in years

    Returns: emi, total_interest, total_payment
    """
    monthly_rate = annual_rate / (12 * 100)
    tenure_months = tenure_years * 12

    emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    return emi, total_interest, total_payment

# Load the trained model for eligibility
model_path = "loan_model.pkl"
if not os.path.exists(model_path):
    st.error("Model not found. Please train it using train_model.py.")
    st.stop()
eligibility_model = joblib.load(model_path)

# --- XGBoostLoanRecommendationEngine CLASS AND FUNCTIONS ---
class XGBoostLoanRecommendationEngine:
    def __init__(self):
        self.suitability_model = None
        self.le_loan = LabelEncoder()
        self.le_bank = LabelEncoder()
        self.df = None
        
    def load_and_prepare_data(self, csv_path="banks_loan_offers.csv"):
        """Load and prepare the loan data"""
        self.df = pd.read_csv(csv_path)
        
        # Encode categorical variables
        self.df['LoanTypeEncoded'] = self.le_loan.fit_transform(self.df['LoanType'])
        self.df['BankEncoded'] = self.le_bank.fit_transform(self.df['BankName'])
        
        return self.df
    
    def train_suitability_model(self):
        """Train XGBoost model to predict loan suitability score"""
        # Create synthetic training data based on loan offers
        training_data = []
        
        for idx, loan_offer in self.df.iterrows():
            # Generate multiple user scenarios for each loan offer
            for salary_multiplier in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
                for age in [25, 30, 35, 40, 45, 50]:
                    for cibil in [650, 700, 750, 800, 850]:
                        # Calculate synthetic user salary based on minimum requirement
                        user_salary = loan_offer['MinSalary'] * salary_multiplier
                        max_affordable_loan = min(loan_offer['MaxLoanAmount'], user_salary * 12 * 5)  # 5x annual income
                        
                        # Calculate suitability score (80-90 range)
                        # Base score starts at 80
                        base_score = 80
                        
                        # Interest rate component (lower rate = higher score)
                        interest_component = max(0, 15 - loan_offer['InterestRate'])
                        
                        # Loan term component (shorter term = higher score)
                        term_component = max(0, 3 - loan_offer['LoanTerm'] * 0.1)
                        
                        # Combined suitability score (80-90 range)
                        suitability_score = base_score + interest_component + term_component
                        
                        # Cap at 90
                        suitability_score = min(90, suitability_score)
                        
                        # Create feature vector
                        features = {
                            'UserSalary': user_salary,
                            'UserAge': age,
                            'UserCIBIL': cibil,
                            'LoanAmount': max_affordable_loan * 0.8,  # Conservative loan amount
                            'MinSalaryReq': loan_offer['MinSalary'],
                            'MaxLoanAmount': loan_offer['MaxLoanAmount'],
                            'InterestRate': loan_offer['InterestRate'],
                            'LoanTerm': loan_offer['LoanTerm'],
                            'BankEncoded': loan_offer['BankEncoded'],
                            'LoanTypeEncoded': loan_offer['LoanTypeEncoded'],
                            'SuitabilityScore': suitability_score
                        }
                        
                        training_data.append(features)
        
        # Convert to DataFrame
        train_df = pd.DataFrame(training_data)
        
        # Prepare features and target
        feature_cols = ['UserSalary', 'UserAge', 'UserCIBIL', 'LoanAmount', 
                        'MinSalaryReq', 'MaxLoanAmount', 'InterestRate', 'LoanTerm',
                        'BankEncoded', 'LoanTypeEncoded']
        
        X = train_df[feature_cols]
        y = train_df['SuitabilityScore']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost regressor for suitability scoring
        self.suitability_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='rmse'
        )
        
        self.suitability_model.fit(X_train, y_train)
        
        return self.suitability_model
    
    def get_personalized_recommendations(self, user_salary, user_age, user_cibil, 
                                         loan_amount, loan_type, top_n=5):
        """Get personalized loan recommendations using XGBoost models"""
        
        # Map loan type to encoded value
        try:
            loan_type_encoded = self.le_loan.transform([loan_type])[0]
        except ValueError:
            # Find the closest match if direct transform fails
            matched_type = None
            for known_type in self.le_loan.classes_:
                if loan_type.lower() in known_type.lower() or known_type.lower() in loan_type.lower():
                    matched_type = known_type
                    break
            if matched_type:
                loan_type_encoded = self.le_loan.transform([matched_type])[0]
            else:
                return pd.DataFrame(), "No matching loan type found in our database for your request."
        
        # Filter loans based on basic eligibility
        eligible_loans = self.df[
            (self.df['MinSalary'] <= user_salary) & 
            (self.df['LoanTypeEncoded'] == loan_type_encoded) & # Use encoded type for filtering
            (self.df['MaxLoanAmount'] >= loan_amount)
        ].copy()
        
        if eligible_loans.empty:
            return pd.DataFrame(), "No loans found matching your criteria"
        
        # Prepare features for each loan offer
        recommendations = []
        
        for idx, loan in eligible_loans.iterrows():
            # Create feature vector for suitability prediction
            features = np.array([[
                user_salary,
                user_age, 
                user_cibil,
                loan_amount,
                loan['MinSalary'],
                loan['MaxLoanAmount'],
                loan['InterestRate'],
                loan['LoanTerm'],
                loan['BankEncoded'],
                loan['LoanTypeEncoded']
            ]])
            
            # Predict suitability score
            suitability_score = self.suitability_model.predict(features)[0]
            
            recommendations.append({
                'BankName': loan['BankName'],
                'LoanType': loan['LoanType'],
                'InterestRate': loan['InterestRate'],
                'MaxLoanAmount': loan['MaxLoanAmount'],
                'LoanTerm': loan['LoanTerm'],
                'MinSalary': loan['MinSalary'],
                'RecommendationScore': round(suitability_score, 2)
            })
        
        # Convert to DataFrame and sort by recommendation score
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('RecommendationScore', ascending=False)
        
        return recommendations_df.head(top_n), "Success"

# Initialize and train the recommendation engine
@st.cache_resource
def initialize_and_train_models(csv_path="banks_loan_offers.csv"):
    """Initialize and train the XGBoost recommendation engine"""
    engine = XGBoostLoanRecommendationEngine()
    engine.load_and_prepare_data(csv_path)
    engine.train_suitability_model()
    return engine

# Function to get recommendations (for integration)
def get_xgboost_recommendations(engine, user_salary, user_age, user_cibil, 
                                 loan_amount, loan_type, top_n=5):
    return engine.get_personalized_recommendations(
        user_salary, user_age, user_cibil, loan_amount, loan_type, top_n
    )

# Initialize recommendation engine
loan_recommendation_engine = initialize_and_train_models("banks_loan_offers.csv")

# ------------------ STYLING  --------------------
st.markdown("""
<style>
/* ---------------- BACKGROUND--------------------*/
.stApp {
   background-color: #f0d4fc;
 font-family: 'Georgia', serif;
   min-height: 100vh;
   padding: 20px;
}

/*------------------TITLE BOX --------------------*/
.title-box {
 background: linear-gradient(to left, #deabf5, #e7c9f5);
 backdrop-filter: blur(30px);
 border-radius: 20px;
 padding: 50px 100px;
 margin-top: 80px;
 max-width: 800px;
 margin-left: auto;
 margin-right: auto;
 box-shadow: 0 0 20px rgba(236, 218, 245, 0.3);
 text-align: center;
 animation: fadeInScale 0.7s ease-out;
}

/*------------------TITLES ON START PAGE --------------------*/
h1.main-title {
 font-family: 'Playfair Display', serif;
 font-size: 65px;
 color: #5c0585;
 margin: 0;
 animation: fadeInScale 0.7s ease-out;
}

h3.subtitle {
 font-family: 'Playfair Display', serif;
 font-size: 25px;
 font-weight: normal;
 color: #5c0585;
 animation: fadeInScale 0.7s ease-out;
}

/*------------------ FORM INPUTS --------------------*/
input, select, textarea {
   border-radius: 10px !important;
   padding: 10px !important;
}

/*------------------ BUTTONS --------------------*/
div.stButton > button,
div.stForm button[type="submit"] {
   background: #deabf5!important;
   color: #5c0585 !important;
   font-weight: bold;
   font-size: 19px;
   border: 2px solid #ecdaf5;
   padding: 10px 40px;
   border-radius: 30px;
   box-shadow: 0px 4px 15px rgba(236, 218, 245, 0.3);
   transition: background-color 0.3s ease, transform 0.2s ease;
   cursor: pointer;
   animation: fadeInScale 0.7s ease-out;
}

div.stButton > button:hover,
div.stForm button[type="submit"]:hover {
   background-color: #ecdaf5 !important;
   transform: scale(1.05);
}

div.stButton,
div.stForm {
   text-align: center;
   margin-top: 30px;
}

label {
   font-family: 'Playfair Display', serif;
   font-weight: normal;
   color: #7507a8 !important;
   font-size: 16px !important;
}

/*------------------STATUS BUTTONS--------------------*/
.status-dot {
   height: 20px;
   width: 20px;
   border-radius: 50%;
   display: inline-block;
   margin-right: 10px;
}

.status-green { background-color: #0ac44e; } 
.status-red { background-color: #ed1f41; }  

/*------------------ANIMATIONS--------------------*/
@keyframes fadeInScale {
   0% {
       opacity: 0;
       transform: scale(0.8);
   }
   100% {
       opacity: 1;
       transform: scale(1);
   }
}

/*------------------CHATBOT STYLES--------------------*/
.chat-container {
    background: linear-gradient(to right, #deabf5, #e7c9f5);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 8px rgba(236, 218, 245, 0.3);
    max-height: 500px;
    overflow-y: auto;
}

.bot-message {
    background: #fff0f9;
    border: 1px solid #deabf5;
    border-radius: 15px 15px 15px 5px;
    padding: 15px;
    margin: 10px 0;
    color: #5c0585;
    font-size: 16px;
    max-width: 80%;
}

.user-message {
    background: #5c0585;
    color: white;
    border-radius: 15px 15px 5px 15px;
    padding: 15px;
    margin: 10px 0 10px auto;
    font-size: 16px;
    max-width: 80%;
    text-align: right;
}

.user-profile {
    background: #fff0f9;
    border: 2px solid #deabf5;
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    color: #5c0585;
}

.progress-bar {
    background: #fff0f9;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}

.progress-fill {
    background: linear-gradient(to right, #5c0585, #deabf5);
    height: 10px;
    border-radius: 5px;
    transition: width 0.3s ease;
}

.warning-box {
    background: #fff3cd;
    border: 2px solid #ffeaa7;
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIMPLIFIED CHATBOT LOGIC --------------------
class SimplifiedLoanChatbot:
    def __init__(self):
        self.questions = [
            {
                "key": "pan",
                "question": "👋 Hi! I'm your loan assistant. Please enter your PAN number to get started (format: ABCDE1234F):",
                "validation": self.validate_pan,
                "error_msg": "Please enter a valid PAN number in format ABCDE1234F"
            },
            {
                "key": "loan_amount",
                "question": "How much loan amount do you need? (minimum ₹1,00,000)",
                "validation": self.validate_loan_amount,
                "error_msg": "Please enter a valid loan amount (minimum ₹1,00,000)"
            },
            {
                "key": "loan_type",
                "question": "What type of loan do you need?\n1. Personal\n2. Home\n3. Education\n4. Business\n\nPlease type the number or name:",
                "validation": self.validate_loan_type,
                "error_msg": "Please select a valid option (1-4 or Personal/Home/Education/Business)"
            }
        ]
        self.current_question = 0
        self.responses = {}
        self.user_profile = None

    def validate_pan(self, value):
        pan_regex = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
        if re.match(pan_regex, value.upper()):
            # Check if PAN exists in database
            user_data = get_user_by_pan(value.upper())
            if user_data:
                self.user_profile = user_data
                return True, value.upper()
            else:
                return False, "PAN number not found in our database. Please contact support or check Admin Panel to add your details."
        return False, None

    def validate_loan_amount(self, value):
        try:
            amount = int(value)
            return amount >= 100000, amount
        except:
            return False, None

    def validate_loan_type(self, value):
        loan_types = {
            "1": "Personal", "personal": "Personal",
            "2": "Home", "home": "Home",
            "3": "Education", "education": "Education",
            "4": "Business", "business": "Business"
        }
        key = value.lower()
        if key in loan_types:
            return True, loan_types[key]
        return False, None

    def get_current_question(self):
        if self.current_question < len(self.questions):
            return self.questions[self.current_question]
        return None

    def process_response(self, user_input):
        current_q = self.get_current_question()
        if not current_q:
            return False, "All questions completed!"

        is_valid, processed_value = current_q["validation"](user_input)
        
        if is_valid:
            self.responses[current_q["key"]] = processed_value
            self.current_question += 1
            
            # Special handling for PAN validation
            if current_q["key"] == "pan" and self.user_profile:
                return True, f"✅ Welcome back, {self.user_profile['name']}! Your details have been loaded from our database."
            else:
                return True, f"✅ Got it! {current_q['key'].replace('_', ' ').title()}: {processed_value}"
        else:
            # Handle special case for PAN not found
            if current_q["key"] == "pan" and isinstance(processed_value, str):
                return False, processed_value
            return False, current_q["error_msg"]

    def is_complete(self):
        return self.current_question >= len(self.questions)

    def get_progress(self):
        return (self.current_question / len(self.questions)) * 100

# ------------------ PAGE MANAGEMENT --------------------
if 'page' not in st.session_state:
   st.session_state.page = 'start'

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = SimplifiedLoanChatbot()

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'awaiting_input' not in st.session_state:
    st.session_state.awaiting_input = False

def go_to_form():
   st.session_state.page = 'form'

def reset_form():
   st.session_state.page = 'form'
   # Clear all session state variables
   for key in list(st.session_state.keys()):
       if key != 'page':
           del st.session_state[key]
   st.session_state.chatbot = SimplifiedLoanChatbot()
   st.session_state.chat_messages = []
   st.session_state.awaiting_input = False

def go_to_recommendations():
   st.session_state.page = 'recommendations'

def go_to_detailed_analysis():
    st.session_state.page = 'detailed_analysis'

# ------------------ STARTING PAGE --------------------
if st.session_state.page == 'start':
   st.markdown("""
   <div class="title-box">
     <h1 class="main-title">LoanEase</h1>
     <h3 class="subtitle">loan recommendation & eligibility</h3>
   </div>
   """, unsafe_allow_html=True)

   col1, col2, col3 = st.columns([2, 1, 2])
   with col2:
       if st.button("Start", key="center_start"):
           st.session_state.page = "form"

# ------------------ CHATBOT FORM PAGE --------------------
elif st.session_state.page == 'form':
    st.markdown("""
    <h1 style="
        font-family: 'Playfair Display', serif;
        font-size: 50px;
        color: #5c0585 ;
        margin-bottom: 20px;
        text-align:center;
        ">
        Quick Loan Application
    </h1>
    """, unsafe_allow_html=True)

    # Progress bar
    progress = st.session_state.chatbot.get_progress()
    st.markdown(f"""
    <div class="progress-bar">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #5c0585; font-weight: bold;">Progress</span>
            <span style="color: #5c0585; font-weight: bold;">{progress:.0f}%</span>
        </div>
        <div style="background: #fff0f9; border-radius: 5px; height: 10px;">
            <div class="progress-fill" style="width: {progress}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show user profile if loaded
    if st.session_state.chatbot.user_profile:
        profile = st.session_state.chatbot.user_profile
        st.markdown(f"""
        <div class="user-profile">
            <h3>👤 Your Profile</h3>
            <strong>Name:</strong> {profile['name']}<br>
            <strong>Age:</strong> {profile['age']}<br>
            <strong>Annual Income:</strong> ₹{profile['annual_income']:,}<br>
            <strong>CIBIL Score:</strong> {profile['cibil_score']}<br>
            <strong>Previous Loan Status:</strong> {profile['previous_loan_status']}
        </div>
        """, unsafe_allow_html=True)

    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Initialize chat if empty
        if not st.session_state.chat_messages and not st.session_state.awaiting_input:
            current_q = st.session_state.chatbot.get_current_question()
            if current_q:
                st.session_state.chat_messages.append(("bot", current_q["question"]))
                st.session_state.awaiting_input = True
        
        # Display chat messages
        for sender, message in st.session_state.chat_messages:
            if sender == "bot":
                st.markdown(f'<div class="bot-message">🤖 {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-message">{message} 👤</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Input section
    if st.session_state.awaiting_input and not st.session_state.chatbot.is_complete():
        user_input = st.text_input("Your response:", key=f"input_{len(st.session_state.chat_messages)}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Send", key="send_message"):
                if user_input.strip():
                    # Add user message to chat
                    st.session_state.chat_messages.append(("user", user_input))
                    
                    # Process the response
                    success, bot_response = st.session_state.chatbot.process_response(user_input)
                    
                    # Add bot response
                    st.session_state.chat_messages.append(("bot", bot_response))
                    
                    if success and not st.session_state.chatbot.is_complete():
                        # Ask next question
                        current_q = st.session_state.chatbot.get_current_question()
                        if current_q:
                            st.session_state.chat_messages.append(("bot", current_q["question"]))
                    elif st.session_state.chatbot.is_complete():
                        # All questions completed
                        st.session_state.chat_messages.append(("bot", "🎉 Perfect! I have all the information I need. Let me analyze your loan eligibility..."))
                        st.session_state.awaiting_input = False
                        
                        # Store responses in session state using database data
                        profile = st.session_state.chatbot.user_profile
                        responses = st.session_state.chatbot.responses
                        
                        st.session_state.pan = responses.get('pan', '')
                        st.session_state.name = profile['name']
                        st.session_state.age = profile['age']
                        st.session_state.income = profile['annual_income']
                        st.session_state.cibil = profile['cibil_score']
                        st.session_state.loan_amount = responses.get('loan_amount', 0)
                        st.session_state.loan_type = responses.get('loan_type', '')
                        st.session_state.prev_loan = profile['previous_loan_status']
                        
                        # Auto-redirect to results after a short delay
                        time.sleep(1)
                        st.session_state.page = 'result'
                    
                    st.rerun()

    # Show completion message and button if chat is complete
    if st.session_state.chatbot.is_complete() and not st.session_state.awaiting_input:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Get My Results", key="get_results"):
                st.session_state.page = 'result'

    # Reset button
    if st.button("Start Over", key="reset_chat"):
        reset_form()

# ------------------ RESULT PAGE --------------------
elif st.session_state.page == 'result':
   age = st.session_state.age
   income = st.session_state.income
   cibil = st.session_state.cibil
   loan_amount = st.session_state.loan_amount
   prev_loan = st.session_state.prev_loan

   # Show user details
   st.markdown(f"""
   <div class="user-profile">
       <h3>👤 Application Details for {st.session_state.name}</h3>
       <strong>PAN:</strong> {st.session_state.pan}<br>
       <strong>Requested Loan Amount:</strong> ₹{loan_amount:,}<br>
       <strong>Loan Type:</strong> {st.session_state.loan_type}
   </div>
   """, unsafe_allow_html=True)

   # Auto rejection rules
   approved = False
   
   if prev_loan == "Ongoing":
       status_class = "status-red"
       message = "❌ NOT APPROVED"
       suggestion = "You have an ongoing loan. New loans cannot be approved."
   elif income < 150000 and loan_amount > 1000000:
       status_class = "status-red"
       message = "❌ NOT APPROVED"
       suggestion = "Requested loan amount is too high for your current income level."
   elif cibil < 600:
       status_class = "status-red"
       message = "❌ NOT APPROVED"
       suggestion = "CIBIL score too low. Minimum 600 is required."
   else:
       # Use the eligibility model for prediction
       monthly_income = income / 12
       est_emi = loan_amount / 60
       emi_bounces = 0 if prev_loan in ["Cleared", "Not Applicable"] else 1
       prev_loan_map = {"Cleared": 0, "Ongoing": 1, "Defaulted": 2, "Not Applicable": 3}
       prev_loan_encoded = prev_loan_map.get(prev_loan, -1)

       input_data = np.array([[age, monthly_income, cibil, est_emi, emi_bounces, prev_loan_encoded, loan_amount]])
       prediction = eligibility_model.predict(input_data)[0]

       if prediction == 1:
           status_class = "status-green"
           message = "✅ LOAN APPROVED"
           suggestion = "Congratulations! You meet the eligibility criteria."
           approved = True
       else:
           status_class = "status-red"
           message = "❌ NOT APPROVED"
           suggestion = "You do not meet the eligibility criteria at this time."

   st.markdown(f"""
   <div style="text-align: center; margin-top: 30px;">
       <span class="status-dot {status_class}"></span>
       <span style="font-size: 22px; font-weight: bold ; color: #5c0585 ;">{message}</span>
   </div>
   """, unsafe_allow_html=True)

   st.markdown(f"<p style='text-align:center; margin-top: 10px; font-size: 18px;color:#5c0585;'>{suggestion}</p>", unsafe_allow_html=True)

   # Show buttons
   col1, col2, col3 = st.columns([1, 1, 1])
   
   if approved:
       with col1:
           if st.button("Try Again"):
               reset_form()
       with col3:
           if st.button("View Recommendations"):
               go_to_recommendations()
   else:
       with col2:
           if st.button("Try Again"):
               reset_form()

# ------------------ RECOMMENDATIONS PAGE --------------------
elif st.session_state.page == 'recommendations':
    st.markdown(
        """
        <h1 style="
            font-family: 'Playfair Display', serif;
            font-size: 50px;
            color: #5c0585 ;
            margin-bottom: 20px;
            text-align:center;
            ">
            Smart Loan Recommendations
        </h1>
        <p style="text-align:center; color:#7507a8;">
            <i>Ranked by AI recommendation score (80-90 range) combining interest rates and loan terms</i>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Get recommendations using the XGBoost engine
    monthly_income = st.session_state.income / 12  # Convert annual to monthly
    recommendations_df, status_message = get_xgboost_recommendations(
        loan_recommendation_engine,
        monthly_income,
        st.session_state.age,
        st.session_state.cibil,
        st.session_state.loan_amount,
        st.session_state.loan_type,
        top_n=5
    )

    if status_message == "Success" and not recommendations_df.empty:
        st.success(f"✅ Found {len(recommendations_df)} loan recommendations!")

        # Create the DataFrame to display
        display_df = recommendations_df[
            ['BankName', 'LoanType', 'InterestRate', 'MaxLoanAmount', 'LoanTerm', 'RecommendationScore']
        ].copy()
        display_df.columns = ['Bank', 'Loan Type', 'Interest Rate (%)', 'Max Amount (₹)', 'Term (years)', 'AI Score']

        # Sort display_df by 'Interest Rate (%)' column in ascending order
        display_df = display_df.sort_values(by='Interest Rate (%)', ascending=True)

        # Reset the index to a sequential 0, 1, 2...
        display_df = display_df.reset_index(drop=True)

        # Add a new 'Rank' column based on the new index
        display_df.insert(0, 'Rank', range(1, 1 + len(display_df)))

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.session_state.recommended_loans = recommendations_df  # Store for detailed analysis

                # Display Best Recommended Bank
        best_bank_by_ai = recommendations_df.iloc[0]['BankName']
        best_score = recommendations_df.iloc[0]['RecommendationScore']
        st.markdown(f"""
        <div style="
            background: #fff0f9;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            text-align: center;
            border: 2px solid #deabf5;
            box-shadow: 0 2px 8px rgba(236, 218, 245, 0.2);
        ">
            <h3 style="color: #5c0585; margin: 0;">🏆 Best Recommended Bank: <span style="font-size: 28px; font-weight: bold;">{best_bank_by_ai}</span></h3>
            <p style="color: #7507a8; margin: 5px 0 0;">(With an AI Recommendation Score of {best_score:.2f})</p>
        </div>
        """, unsafe_allow_html=True)

        # Plotting the interest rate vs recommendation score
        st.subheader("📊 Interest Rate vs Recommendation Score")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by interest rate for better visualization
        plot_df = recommendations_df.sort_values('InterestRate')
        
        ax.plot(plot_df['InterestRate'], plot_df['RecommendationScore'], 
                marker='o', linestyle='-', color='#5c0585', alpha=0.7)
        ax.set_xlabel('Interest Rate (%)')
        ax.set_ylabel('Recommendation Score')
        ax.set_title('Interest Rate vs Recommendation Score', fontsize=16, color='#5c0585')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(plot_df['InterestRate'])
        ax.set_xticklabels([f"{ir:.2f}%" for ir in plot_df['InterestRate']], rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)

        # Bank selection dropdown for detailed analysis
        st.subheader("🏦 Choose a Bank for Detailed Analysis:")
        bank_options = [''] + sorted(list(recommendations_df['BankName'].unique()))
        selected_bank_for_detail = st.selectbox("Select Bank:", bank_options, key='select_bank_detail')

        if selected_bank_for_detail:
            # Store the selected bank name for use in the next page
            st.session_state.selected_bank_for_detail = selected_bank_for_detail
            # Add a button to navigate to detailed analysis
            if st.button(f"View Detailed Analysis for {selected_bank_for_detail}"):
                go_to_detailed_analysis()
                st.rerun()

    else:
        st.error(f"😞 {status_message}")
        st.info("Please adjust your loan requirements or profile details and try again.")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back to Eligibility Result"):
            st.session_state.page = 'result'
            st.rerun()
    with col3:
        if st.button("Start Over"):
            reset_form()
            st.rerun()

# ------------------ DETAILED ANALYSIS PAGE --------------------
elif st.session_state.page == 'detailed_analysis':
    st.markdown("""
    <h1 style="
        font-family: 'Playfair Display', serif;
        font-size: 50px;
        color: #5c0585 ;
        margin-bottom: 20px;
        text-align:center;
        ">
        Detailed Loan Analysis
    </h1>
    """, unsafe_allow_html=True)

    # Show user's personal details
    st.markdown(f"""
    <div style="
        background: #f9f3fd;
        border-radius: 10px;
        padding: 20px;
        color: #5c0585;
        font-size: 18px;">
        <strong>Requested Loan Amount:</strong> ₹{st.session_state.loan_amount:,.0f}<br>
        <strong>Requested Loan Type:</strong> {st.session_state.loan_type}<br>
        <strong>Your Age:</strong> {st.session_state.age}<br>
        <strong>Your Annual Income:</strong> ₹{st.session_state.income:,.0f}<br>
        <strong>Your CIBIL Score:</strong> {st.session_state.cibil}
    </div>
    """, unsafe_allow_html=True)

    selected_bank_name = st.session_state.get('selected_bank_for_detail', '')

    if selected_bank_name and 'recommended_loans' in st.session_state and not st.session_state.recommended_loans.empty:
        # Filter for the selected bank's loan offer
        selected_loan = st.session_state.recommended_loans[
            st.session_state.recommended_loans['BankName'] == selected_bank_name
        ].iloc[0]

        st.success(f"✅ You are viewing details for **{selected_loan['BankName']}** (AI Score: {selected_loan['RecommendationScore']:.2f})")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Loan Offer Details")
            st.markdown(f"- **Loan Type:** {selected_loan['LoanType']}")
            st.markdown(f"- **Interest Rate:** {selected_loan['InterestRate']}%")
            st.markdown(f"- **Maximum Loan Amount:** ₹{selected_loan['MaxLoanAmount']:,}")
            st.markdown(f"- **Loan Term:** {selected_loan['LoanTerm']} years")
            st.markdown(f"- **Minimum Salary Required:** ₹{selected_loan['MinSalary']:,}")

        with col2:
            st.markdown("### 🎯 AI Analysis")
            st.markdown(f"- **AI Recommendation Score:** {selected_loan['RecommendationScore']:.2f}")
            
            # Find rank based on RecommendationScore
            rank = st.session_state.recommended_loans['RecommendationScore'].rank(ascending=False).loc[selected_loan.name]
            st.markdown(f"- **Rank among recommendations:** #{int(rank)} out of {len(st.session_state.recommended_loans)}")

        # Loan payment calculator
        st.markdown("### 📊 Loan Payment Calculator")
        
        # Let user adjust tenure
        tenure_years = st.slider(
            "Adjust loan tenure (years):",
            min_value=1,
            max_value=30,
            value=int(selected_loan['LoanTerm']),
            step=1
        )

        # Calculate EMI, total interest, and total payment
        emi, total_interest, total_payment = calculate_emi(
            principal=st.session_state.loan_amount,
            annual_rate=selected_loan['InterestRate'],
            tenure_years=tenure_years
        )
        
        st.markdown(f"**Monthly EMI:** ₹{emi:,.2f}")
        st.markdown(f"**Total Amount Payable:** ₹{total_payment:,.2f}")
        st.markdown(f"**Total Interest:** ₹{total_interest:,.2f}")

        # Pie chart for principal vs interest breakdown
        fig, ax = plt.subplots()
        ax.pie(
                        [st.session_state.loan_amount, total_interest],
            labels=["Principal", "Interest"],
            colors=["#deabf5", "#e7c9f5"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Principal vs. Interest Breakdown", fontsize=14, color="#5c0585")
        st.pyplot(fig)

        # Download button for summary
        summary_text = f"""
Loan Summary for {selected_loan['BankName']}:

Loan Amount: ₹{st.session_state.loan_amount:,.0f}
Interest Rate: {selected_loan['InterestRate']}%
Loan Term: {tenure_years} years

Calculated EMI: ₹{emi:,.0f}/month
Total Interest: ₹{total_interest:,.0f}
Total Payment: ₹{total_payment:,.0f}

AI Recommendation Score: {selected_loan['RecommendationScore']:.2f}
"""
        st.download_button(
            label=f"Download {selected_loan['BankName']} Summary",
            data=summary_text,
            file_name=f"{selected_loan['BankName']}_detailed_summary.txt",
            mime="text/plain"
        )

    else:
        st.error("Please go back to recommendations and select a bank for detailed analysis.")

    # Navigation back to recommendations
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Recommendations"):
            st.session_state.page = 'recommendations'
            st.rerun()
    with col2:
        if st.button("Go to Start"):
            reset_form()
            st.rerun()
