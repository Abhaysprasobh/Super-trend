"use client";
import React, { useState } from 'react';
import axios from 'axios';
import dynamic from 'next/dynamic';

// Dynamically import the useRouter hook to avoid SSR errors
const Router = dynamic(() => import('next/router'), { ssr: false });

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    // Simple client-side validation for email and password
    // if (!/\S+@\S+\.\S+/.test(email)) {
    //   setError('Please enter a valid email address.');
    //   setIsLoading(false);
    //   return;
    // }
    // if (password.length < 6) {
    //   setError('Password must be at least 6 characters long.');
    //   setIsLoading(false);
    //   return;
    // }

    try {
      const response = await axios.post('http://127.0.0.1:5000/api/login', {
        username:email,
        password,
      });

      const token = response.data.token;
      localStorage.setItem('authToken', token);

      alert('Login successful!');
      Router.push('/dashboard'); // Modify the route as per your application
    } catch (err) {
      setError('Invalid email or password. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <section className="w-full max-w-md px-6 py-12 bg-white shadow-lg rounded-lg">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-cyan-600">Login</h1>
        </div>

        <form onSubmit={handleLogin} className="mt-8 space-y-6 flex flex-col items-center">
          {/* Email Input */}
          <div className="w-full">
            <label htmlFor="email" className="sr-only">Email:</label>
            <input
              type="text"
              id="email"
              className="w-full border border-gray-300 p-4 rounded-lg shadow-sm text-sm focus:ring-cyan-500 focus:border-green-500"
              placeholder="Enter your username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          {/* Password Input */}
          <div className="w-full relative">
            <label htmlFor="password" className="sr-only">Password:</label>
            <input
              type={showPassword ? 'text' : 'password'}
              id="password"
              className="w-full border border-gray-300 p-4 rounded-lg shadow-sm text-sm focus:ring-cyan-500 focus:border-green-500"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            {/* Toggle Password Visibility */}
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-500"
            >
              {showPassword ? 'Hide' : 'Show'}
            </button>
          </div>

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              className={`bg-cyan-600 text-white py-2 px-6 rounded-lg hover:bg-cyan-700 focus:ring focus:ring-green-500 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={isLoading}
            >
              {isLoading ? (
                <span className="spinner-border animate-spin h-5 w-5 border-t-2 border-white rounded-full"></span>
              ) : (
                'Sign in'
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && <p className="text-red-600 text-sm text-center">{error}</p>}

          {/* Register Link */}
          <div className="flex justify-center items-center">
            <p className="text-sm text-gray-500">
              No account?
              <a href="../Register" className="text-cyan-600 hover:underline ml-1">Sign up</a>
            </p>
          </div>
        </form>
      </section>
    </div>
  );
}

export default Login;
