"use client";
import React, { useState } from 'react';
import axios from 'axios';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(''); 

    try {
      const response = await axios.post('https://jassim.com/login', {
        email,
        password,
      });

      const token = response.data.token;
      localStorage.setItem('authToken', token);
      alert('Login successful!');
    } catch (err) {
      setError('Invalid email or password. Please try again.');
    }
  };

  return (
    <div className="min-h-screen  flex items-center justify-center">
      <section className="w-full max-w-md px-6 py-12 bg-white shadow-lg rounded-lg">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-red-600">Login</h1>
        </div>

        <form onSubmit={handleLogin} className="mt-8 space-y-6">

          
          <div>
            <label htmlFor="email" className="sr-only">Email:</label>
            <input
              type="email"
              id="email"
              className="w-full border-gray-300 p-4 rounded-lg shadow-sm text-sm focus:ring-red-500 focus:border-red-500"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="sr-only">Password:</label>
            <input
              type="password"
              id="password"
              className="w-full border-gray-300 p-4 rounded-lg shadow-sm text-sm focus:ring-red-500 focus:border-red-500"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

        <div>
          <button
              type="submit"
              className="bg-green-600 text-white py-2 px-6 rounded-lg hover:bg-green-700 focus:ring focus:ring-green-500"
            >
              Sign in
          </button>
        </div>

        {error && <p className="text-red-600 text-sm text-center">{error}</p>}


          <div className="flex justify-center items-center">
            <p className="text-sm text-gray-500">
              No account?
              <a href="../Register" className="text-red-600 hover:underline ml-1">Sign up</a>
            </p>
          </div>
        </form>
      </section>
    </div>
  );
}

export default Login;
