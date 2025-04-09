import requests
import json

BASE_URL = "http://127.0.0.1:5000/api"

def print_response(response):
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print("Raw Response:")
        print(response.text)
    print("-" * 50)

def test_register():
    print("Testing User Registration...")
    response = requests.post(f"{BASE_URL}/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass"
    })
    print_response(response)

def test_login():
    print("Testing User Login...")
    response = requests.post(f"{BASE_URL}/login", json={
        "username": "testuser",
        "password": "testpass"
    })
    print_response(response)
    if response.status_code == 200:
        token = response.json().get("token")
        return token
    return None

def test_get_user(token):
    print("Testing Get User Info...")
    response = requests.get(f"{BASE_URL}/user", headers={
        "x-access-token": str(token)
    })
    print_response(response)

def test_get_stock_data(token, ticker="AAPL"):
    print(f"Testing Get Stock Data for {ticker}...")
    response = requests.get(f"{BASE_URL}/stock", headers={
        "x-access-token": str(token)
    }, params={
        "ticker": ticker
    })
    print_response(response)


def test_get_indicator_comparison(token, symbol="AAPL", interval="1d", days=700):
    print(f"Testing Get Indicator Comparison for {symbol}...")
    response = requests.post(f"{BASE_URL}/indicator", headers={
        "x-access-token": str(token)
    }, json={
        "symbol": symbol,
        "interval": interval,
        "days": days
    })
    print_response(response)


def test_save_settings(token):
    print("Testing Save User Settings...")
    response = requests.post(f"{BASE_URL}/settings", headers={
        "x-access-token": str(token)
    }, json={
        "ticker": "AAPL",
        "atr_len": 10,
        "factor": 3.0,
        "training_data_period": 100,
        "highvol": 0.75,
        "midvol": 0.5,
        "lowvol": 0.25,
        "high_multiplier": 2.0,
        "mid_multiplier": 3.0,
        "low_multiplier": 4.0,
        "days": 700
    })
    print_response(response)

def test_get_notifications(token):
    print("Testing Get Notifications...")
    response = requests.get(f"{BASE_URL}/notifications", headers={
        "x-access-token": str(token)
    })
    print_response(response)
def test_adaptive_supertrend(token):
    print("Testing Adaptive SuperTrend...")
    payload = {
        "ticker": "AAPL",
        "atr_len": 10,
        "factor": 3.0,
        "training_data_period": 100,
        "highvol": 0.75,
        "midvol": 0.5,
        "lowvol": 0.25,
        "high_multiplier": 2.0,
        "mid_multiplier": 3.0,
        "low_multiplier": 4.0,
        "days": 700
    }

    response = requests.post(f"{BASE_URL}/adaptive", headers={
        "x-access-token": str(token)
    }, json=payload)
    print_response(response)
def test_basic_supertrend(ticker="AAPL", timerange="1mo", length=10, multiplier=3.0):
    print("Testing Basic SuperTrend...")
    response = requests.get(f"{BASE_URL}/supertrend", params={
        "ticker": ticker,
        "range": timerange,
        "length": length,
        "multiplier": multiplier
    })
    print_response(response)


if __name__ == "__main__":
    print("Starting API Integration Tests")
    print("=" * 50)
    test_register()
    token = test_login()
    if token:
        # test_get_user(token)
        # test_get_stock_data(token, "AAPL")
        # test_get_stock_data(token, "GOOGL")
        # test_get_indicator_comparison(token, symbol="AAPL", interval="1d", days=700)
        # test_save_settings(token)
        # test_get_notifications(token)
        # test_basic_supertrend("AAPL")
        # test_adaptive_supertrend(token)
    else:
        print("Login failed, skipping further tests.")
