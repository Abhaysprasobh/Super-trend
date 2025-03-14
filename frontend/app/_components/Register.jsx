"use client";
import React, { useState } from 'react';
import axios from 'axios';

function Register() {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    passwordConfirmation: '',
    marketingAccept: false
  });

  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (formData.password !== formData.passwordConfirmation) {
      setError("Passwords do not match.");
      return;
    }

    try {
      const response = await axios.post('http://127.0.0.1:5000/api/register', {
        username:formData.username,
        email:formData.email,
        password:formData.password,
      });
      console.log(response);
      alert('Account created successfully!');
    } catch (err) {
      setError('An error occurred while creating the account. Please try again.');
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
      <section className="bg-white w-[90vw] max-w-[480px] rounded-lg shadow-lg p-8">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-cyan-600 mb-6">Registration</h1>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && <p className="text-red-600 text-sm text-center">{error}</p>}

          <div className="grid grid-cols-1 gap-4">
            <input
              type="text"
              id="username"
              name="username"
              placeholder="Username"
              value={formData.username}
              onChange={handleInputChange}
              className="w-full h-12 p-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            />

            {/* <input
              type="text"
              id="lastName"
              name="lastName"
              placeholder="Last Name"
              value={formData.lastName}
              onChange={handleInputChange}
              className="w-full h-12 p-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            /> */}

            <input
              type="email"
              id="email"
              name="email"
              placeholder="Email"
              value={formData.email}
              onChange={handleInputChange}
              className="w-full h-12 p-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            />

            <input
              type="password"
              id="password"
              name="password"
              placeholder="Password"
              value={formData.password}
              onChange={handleInputChange}
              className="w-full h-12 p-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            />

            <input
              type="password"
              id="passwordConfirmation"
              name="passwordConfirmation"
              placeholder="Confirm Password"
              value={formData.passwordConfirmation}
              onChange={handleInputChange}
              className="w-full h-12 p-3 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            />

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="marketingAccept"
                name="marketingAccept"
                checked={formData.marketingAccept}
                onChange={handleInputChange}
                className="h-5 w-5 text-cyan-600"
              />
              <span className="text-sm text-gray-700">
                I want to receive emails about events, product updates, and company announcements.
              </span>
            </div>

            <p className="text-sm text-gray-500 text-center">
              By creating an account, you agree to our
              <a href="#" className="text-cyan-600 underline"> terms and conditions</a> and
              <a href="#" className="text-cyan-600 underline"> privacy policy</a>.
            </p>

            <button
              type="submit"
              className="w-full h-12 bg-cyan-600 text-white font-semibold rounded-md hover:bg-cyan-700 transition duration-200 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              Create an account
            </button>

            <p className="text-sm text-center text-gray-500">
              Already have an account? <a href="../Login" className="text-cyan-600 underline">Log in</a>.
            </p>
          </div>
        </form>
      </section>
    </div>
  );
}

export default Register;
