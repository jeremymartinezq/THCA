import os
from dotenv import load_dotenv
import plaid
import stripe
import requests
import openai
from alchemy_sdk import Alchemy
from alpha_vantage.timeseries import TimeSeries
import quandl
from flask import Flask, request, jsonify

# Load environment variables
load_dotenv()

PLAID_CLIENT_ID = os.getenv('PLAID_CLIENT_ID')
PLAID_SECRET = os.getenv('PLAID_SECRET')
STRIPE_API_KEY = os.getenv('STRIPE_API_KEY')
COMPLY_ADVANTAGE_API_KEY = os.getenv('COMPLY_ADVANTAGE_API_KEY')
JUMIO_API_TOKEN = os.getenv('JUMIO_API_TOKEN')
JUMIO_API_SECRET = os.getenv('JUMIO_API_SECRET')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')

# Plaid API initialization
plaid_client = plaid.Client(client_id=PLAID_CLIENT_ID, secret=PLAID_SECRET, environment='sandbox')

# Stripe API initialization
stripe.api_key = STRIPE_API_KEY

# Alpha Vantage API initialization
ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')

# Alchemy API initialization
alchemy = Alchemy(ALCHEMY_API_KEY)

# Quandl API initialization
quandl.ApiConfig.api_key = QUANDL_API_KEY

# OpenAI API initialization
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the THCa Fintech Platform!"


@app.route('/verify_user', methods=['POST'])
def verify_user():
    user_data = request.json
    document = user_data.get('document')

    # Run KYC check using Jumio
    kyc_result = run_kyc_check(document)
    if not kyc_result.get('success'):
        return jsonify({"status": "failure", "reason": "KYC check failed"}), 400

    # Run AML check using ComplyAdvantage
    aml_result = run_aml_check(user_data['name'])
    if aml_result.get('result') == 'match':
        return jsonify({"status": "failure", "reason": "AML check flagged the user"}), 400

    return jsonify({"status": "success"}), 200


@app.route('/process_payment', methods=['POST'])
def process_payment():
    payment_info = request.json

    # Process payment with Stripe
    payment_intent = stripe.PaymentIntent.create(
        amount=payment_info['amount'],
        currency=payment_info['currency'],
        payment_method_types=['card'],
        description=payment_info['description']
    )

    return jsonify(payment_intent), 200


@app.route('/link_bank', methods=['POST'])
def link_bank():
    public_token = request.json.get('public_token')

    # Exchange public token for access token using Plaid
    exchange_response = plaid_client.Item.public_token.exchange(public_token)
    access_token = exchange_response['access_token']

    # Retrieve account information
    account_info = plaid_client.Auth.get(access_token)

    return jsonify(account_info), 200


@app.route('/market_data', methods=['GET'])
def market_data():
    symbol = request.args.get('symbol', 'THCA')

    # Get market data from Alpha Vantage
    data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')

    # Run AI prediction using OpenAI GPT
    prediction = get_market_data_and_predict(symbol, data)

    return jsonify({"market_data": data.to_dict(), "prediction": prediction}), 200


@app.route('/blockchain_data', methods=['GET'])
def blockchain_data():
    transaction_id = request.args.get('transaction_id')

    # Retrieve blockchain data using Alchemy
    transaction_data = alchemy.core.get_transaction_receipt(transaction_id)

    return jsonify(transaction_data), 200


if __name__ == "__main__":
    app.run(debug=True)


# Helper functions
def run_kyc_check(document):
    response = requests.post(
        "https://netverify.com/api/netverify/v2/initiate",
        auth=(JUMIO_API_TOKEN, JUMIO_API_SECRET),
        json=document
    )
    return response.json()


def run_aml_check(customer_name):
    response = requests.post(
        "https://api.complyadvantage.com/v1/searches",
        headers={"Authorization": f"Token {COMPLY_ADVANTAGE_API_KEY}"},
        json={"name": customer_name, "type": "individual"}
    )
    return response.json()


def get_market_data_and_predict(symbol, data):
    prompt = f"Analyze the market data for {symbol} and predict the next price movement."
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text
