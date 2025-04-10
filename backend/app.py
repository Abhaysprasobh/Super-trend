import os
import json
from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import yfinance as yf
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
from supertrend import get_supertrend_data
from adaptive import get_adaptive_supertrend_json
from adapt_test import supertrend_strategy_comparison_json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
CORS(app, supports_credentials=True, origins=["http://localhost:3000"]) 

# MySQL configurations
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

mysql = MySQL(app)

def create_tables():
    with app.app_context():
        cur = mysql.connection.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                ticker VARCHAR(20),
                atr_len INT DEFAULT 10,
                factor FLOAT DEFAULT 3.0,
                training_data_period INT DEFAULT 100,
                highvol FLOAT DEFAULT 0.75,
                midvol FLOAT DEFAULT 0.5,
                lowvol FLOAT DEFAULT 0.25,
                high_multiplier FLOAT DEFAULT 2.0,
                mid_multiplier FLOAT DEFAULT 3.0,
                low_multiplier FLOAT DEFAULT 4.0,
                days INT DEFAULT 700,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cur.close()

create_tables()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s", (token,))
        user = cur.fetchone()
        cur.close()

        if not user:
            return jsonify({'message': 'Invalid token!'}), 401

        return f(user, *args, **kwargs)

    return decorated

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = generate_password_hash(data.get('password'))
    print(data)

    if not all([username, email, password]):
        print("missing")
        return jsonify({'message': 'Missing required fields'}), 400

    cur = mysql.connection.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, password)
        )
        mysql.connection.commit()
        print("successfull")
        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        print(e)
        return jsonify({'message': str(e)}), 400
    finally:
        cur.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    print(data)
    username = data.get('username')
    password = data.get('password')

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()

    if not user or not check_password_hash(user[3], password):
        print("not found user")
        return jsonify({'message': 'Invalid credentials'}), 401

    return jsonify({'token': user[0]}), 200

# @app.route('/api/stock', methods=['GET'])
# # @token_required
# def get_stock_data():
#     data = request.get_json()
#     ticker = data.get('ticker')

#     if not ticker:
#         return jsonify({'message': 'Ticker is required'}), 400

#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info
#         history = stock.history(period="1mo").reset_index()
        
#         # Convert DataFrame to JSON serializable format
#         history = history[['Date', 'Close']].to_dict(orient='records')
        
    #     return jsonify({
    #         'ticker': ticker,
    #         'info': {
    #             'currentPrice': info.get('currentPrice'),
    #             'dayHigh': info.get('dayHigh'),
    #             'dayLow': info.get('dayLow')
    #         },
    #         'history': history
    #     }), 200
    # except Exception as e:
    #     return jsonify({'message': str(e)}), 500
    



@app.route('/api/stock', methods=['GET'])
# @token_required
def get_stock_data():
    # Get ticker from query string
    ticker = request.args.get('ticker')
    
    if not ticker:
        return jsonify({'message': 'Ticker is required'}), 400

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1mo").reset_index()
        
        # Convert DataFrame to JSON serializable format
        history = history[['Date', 'Close']].to_dict(orient='records')
        
        return jsonify({
            'ticker': ticker,
            'info': {
                'currentPrice': info.get('currentPrice'),
                'dayHigh': info.get('dayHigh'),
                'dayLow': info.get('dayLow')
            },
            'history': history
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    

# new stuff
@app.route('/api/indicator', methods=['POST'])
def get_indicator_comparison():
    """
    New API endpoint that returns indicator data, backtest performance,
    signals, equity curves, and annual returns for a given symbol.
    """
    data = request.get_json()
    symbol = data.get('symbol')
    high_vol = data.get('high_vol_multiplier', 3)
    mid_vol = data.get('mid_vol_multiplier', 2)
    low_vol = data.get('low_vol_multiplier', 1)
    days = data.get('days', 700)
    
    if not symbol:
        return jsonify({'message': 'Symbol is required'}), 400
    
    try:
        result = supertrend_strategy_comparison_json(
            ticker=symbol,
            days=days,
            high_vol_multiplier=high_vol,
            mid_vol_multiplier=mid_vol,
            low_vol_multiplier=low_vol
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    
@app.route('/api/supertrend', methods=['GET'])
def supertrend_basic():
    ticker = request.args.get('ticker')
    timerange = request.args.get('range', '1mo')
    length = int(request.args.get('length', 7))
    multiplier = float(request.args.get('multiplier', 3.0))

    try:
        response = get_supertrend_data(ticker, timerange, length, multiplier)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/adaptive', methods=['POST'])
@token_required
def adaptive_supertrend(current_user):
    data = request.get_json()
    try:
        response = get_adaptive_supertrend_json(data)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500











@app.route('/api/user', methods=['GET'])
@token_required
def get_user(current_user):
    return jsonify({
        'id': current_user[0],
        'username': current_user[1],
        'email': current_user[2]
    }), 200













if __name__ == '__main__':
    app.run(debug=True)