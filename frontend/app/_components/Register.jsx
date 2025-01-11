"use client";
import React, { useState } from 'react';
import axios from 'axios';
import Image from 'next/image';

function Register() {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
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
      const response = await axios.post('https://jassim.com/register', formData);

      alert('Account created successfully!');

    } catch (err) {
      setError('An error occurred while creating the account. Please try again.');
    }
  };

  return (
    <div>
      <section className="bg-white w-[80vw] max-w-[480px]">
        <div className="lg:grid lg:min-h-screen lg:grid-cols-6">
          <main className="flex items-center justify-center px-8 py-8 sm:px-12 lg:col-span-7 lg:px-16 lg:py-12 xl:col-span-6">
            <div className="max-w-[340px]">
              <div className="relative flex justify-center -mt-16">
                <h1 className="mt-2 text-xl font-bold text-gray-900">
                  Registration
                </h1>
              </div>

              <form onSubmit={handleSubmit} className="mt-8 grid grid-cols-6 gap-6">
                {error && <p className="text-red-600 text-sm col-span-6">{error}</p>}

                <div className="col-span-6 sm:col-span-3">
                  <input
                    type="text"
                    id="firstName"
                    name="firstName"
                    placeholder='   First Name'
                    value={formData.firstName}
                    onChange={handleInputChange}
                    className="mt-1 w-full h-8 rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
                  />
                </div>

                <div className="col-span-6 sm:col-span-3">
                  <input
                    type="text"
                    id="lastName"
                    name="lastName"
                    placeholder='   Last Name'
                    value={formData.lastName}
                    onChange={handleInputChange}
                    className="mt-1 w-full h-8 rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
                  />
                </div>

                <div className="col-span-6">
                  <input
                    type="email"
                    id="email"
                    name="email"
                    placeholder='   Email'
                    value={formData.email}
                    onChange={handleInputChange}
                    className="mt-1 w-full h-8 rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
                  />
                </div>

                <div className="col-span-6 sm:col-span-3">
                  <input
                    type="password"
                    id="password"
                    name="password"
                    placeholder='   Password'
                    value={formData.password}
                    onChange={handleInputChange}
                    className="mt-1 w-full h-8 rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
                  />
                </div>

                <div className="col-span-8 sm:col-span-3">
                  <input
                    type="password"
                    id="passwordConfirmation"
                    name="passwordConfirmation"
                    placeholder='   Confirm Password'
                    value={formData.passwordConfirmation}
                    onChange={handleInputChange}
                    className="mt-1 w-full h-8 rounded-md bg-white text-sm text-gray-700 shadow-sm"
                  />
                </div>

                <div className="col-span-6">
                  <label htmlFor="marketingAccept" className="flex gap-4">
                    <input
                      type="checkbox"
                      id="marketingAccept"
                      name="marketingAccept"
                      checked={formData.marketingAccept}
                      onChange={handleInputChange}
                      className="size-5 rounded-md border-gray-200 bg-white shadow-sm"
                    />
                    <span className="text-sm text-gray-700">
                      I want to receive emails about events, product updates and company announcements.
                    </span>
                  </label>
                </div>

                <div className="col-span-6">
                  <p className="text-sm text-gray-500">
                    By creating an account, you agree to our
                    <a href="#" className="text-gray-700 underline"> terms and conditions </a>
                    and
                    <a href="#" className="text-gray-700 underline">privacy policy</a>.
                  </p>
                </div>

                <div className="col-span-6 sm:flex sm:items-center sm:gap-4">
                  <button
                    type="submit"
                    className="inline-block shrink-0 rounded-md border border-blue-600 bg-blue-600 px-12 py-3 text-sm font-medium text-white transition hover:bg-transparent hover:text-blue-600 focus:outline-none focus:ring active:text-blue-500"
                  >
                    Create an account
                  </button>

                  <p className="mt-4 text-sm text-gray-500 sm:mt-0">
                    Already have an account?
                    <a href="../Login" className="text-gray-700 underline">Log in</a>.
                  </p>
                </div>
              </form>
            </div>
          </main>
        </div>
      </section>
    </div>
  );
}

export default Register;
