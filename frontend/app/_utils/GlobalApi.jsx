import axios from "axios";

const API_BASE_URL = "http://localhost:5000";

export const loginUser = async (email, password) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/login`, { email, password }, { 
      headers: { "Content-Type": "application/json" },
      withCredentials: true, // For secure authentication using cookies
    });

    return response.data;
  } catch (error) {
    throw error.response?.data?.message || "Login failed. Please try again.";
  }
};

export const registerUser = async (userData) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/register`, userData, {
        headers: { "Content-Type": "application/json" },
        withCredentials: true,
      });
  
      return response.data;
    } catch (error) {
      throw error.response?.data?.message || "Registration failed. Try again.";
    }
  };

  
  const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: { "Content-Type": "application/json" },
    withCredentials: true, // Ensures cookies are sent for authentication
  });
  
  const handleApiError = (error) => {
    if (error.response) {
      return error.response.data?.message || "An error occurred. Please try again.";
    } else if (error.request) {
      return "No response from the server. Check your network connection.";
    } else {
      return "Request setup error. Please try again.";
    }
  };
  
// Fetch stock data via GET
export async function fetchStockData(ticker) {
  const response = await fetch(`${API_BASE_URL}/api/stock?ticker=${encodeURIComponent(ticker)}`, {
    method: 'GET'
  });
  if (!response.ok) {
    const errorResponse = await response.json();
    throw new Error(errorResponse.message || 'Error fetching stock data');
  }
  return await response.json();
}

export async function fetchIndicatorComparison(symbol, interval, days) {
  const response = await fetch(`${API_BASE_URL}/api/indicator`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ symbol, interval, days }),
  });

  if (!response.ok) {
    const errorResponse = await response.json();
    throw new Error(errorResponse.message || "Error fetching indicator data");
  }

  return await response.json();
}

  // Fetch user details
  export const getUserData = async () => {
    try {
      const response = await apiClient.get("/user");
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  };
  