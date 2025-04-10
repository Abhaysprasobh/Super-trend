"use client";

import React, { useState } from "react";
import { fetchAdaptiveSupertrend } from "../_utils/GlobalApi"; // Adjust the path based on your folder structure

export default function AdaptiveSupertrendPage() {
  const [formData, setFormData] = useState({
    ticker: "RELIANCE.NS",
    atr_len: 10,
    factor: 3.0,
    training_data_period: 100,
    highvol: 0.75,
    midvol: 0.5,
    lowvol: 0.25,
    high_multiplier: 2.0,
    mid_multiplier: 3.0,
    low_multiplier: 4.0,
    days: 700,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: isNaN(value) ? value : parseFloat(value),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetchAdaptiveSupertrend(formData);
      setResult(response);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-6">Adaptive Supertrend Settings</h1>
      <form onSubmit={handleSubmit} className="bg-white p-6 rounded-2xl shadow-md w-full max-w-2xl grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.keys(formData).map((key) => (
          <div key={key} className="flex flex-col">
            <label htmlFor={key} className="text-sm font-semibold capitalize">
              {key.replace(/_/g, " ")}
            </label>
            <input
              type="text"
              id={key}
              name={key}
              value={formData[key]}
              onChange={handleChange}
              className="p-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </div>
        ))}
        <button
          type="submit"
          className="col-span-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
          disabled={loading}
        >
          {loading ? "Loading..." : "Submit"}
        </button>
      </form>

      {error && <p className="text-red-500 mt-4">{error}</p>}

      {result && (
        <div className="mt-8 w-full max-w-4xl bg-white p-6 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">Response</h2>
          <pre className="text-sm overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
