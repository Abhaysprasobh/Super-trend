import axios from "axios";

const API_BASE_URL = "http://localhost:3000";

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
  
  // const handleApiError = (error) => {
  //   if (error.response) {
  //     return error.response.data?.message || "An error occurred. Please try again.";
  //   } else if (error.request) {
  //     return "No response from the server. Check your network connection.";
  //   } else {
  //     return "Request setup error. Please try again.";
  //   }
  // };
  
  // Fetch stock data
  export const fetchStockData = async (ticker) => {
    try {
      const response = await apiClient.post("/stock", { ticker });
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  };
  
  // Fetch indicator comparison
  export const getIndicatorComparison = async (symbol, interval = "1d", days = 700) => {
    try {
      const response = await apiClient.post("/indicator", { symbol, interval, days });
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  };
  
  // Fetch user details
  export const getUserData = async () => {
    try {
      const response = await apiClient.get("/user");
      return response.data;
    } catch (error) {
      throw handleApiError(error);
    }
  };
  