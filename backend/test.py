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
    response = requests.post(f"{BASE_URL}/stock", headers={
        "x-access-token": str(token)
    }, json={
        "ticker": ticker
    })
    print_response(response)

if __name__ == "__main__":
    print("Starting API Integration Tests")
    print("=" * 50)
    test_register()
    token = test_login()
    if token:
        test_get_user(token)
        test_get_stock_data(token, "AAPL")
        test_get_stock_data(token, "GOOGL")
    else:
        print("Login failed, skipping further tests.")
