"use client";

import React, { useState } from "react";
import { fetchAdaptiveSupertrend } from "../_utils/GlobalApi";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Label,
} from "recharts";

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
      console.log("Response:", response);
      setResult(response);
    } catch (err) {
      console.error(err);
      setError("Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  // Format signals for scatter chart
  const buySignals = result?.signals?.filter((s) => s.signal === "buy");
  const sellSignals = result?.signals?.filter((s) => s.signal === "sell");

  const chartData = result?.signals || [];

  const equityCurve = result?.equity_curve?.map((value, index) => ({
    index,
    capital: value,
  }));

  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-6">Adaptive Supertrend Strategy</h1>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded-2xl shadow-md w-full max-w-3xl grid grid-cols-1 md:grid-cols-2 gap-4"
      >
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

      {/* Error */}
      {error && <p className="text-red-500 mt-4">{error}</p>}

      {/* Stats */}
      {result && (
        <div className="mt-6 text-center bg-white p-4 rounded-lg shadow w-full max-w-2xl">
          <p className="text-lg">
            <strong>Final Capital:</strong> ₹{result.final_capital}
          </p>
          <p className="text-lg">
            <strong>Annual Return:</strong> {(result.annual_return * 100).toFixed(2)}%
          </p>
        </div>
      )}

      {/* Price Chart with Buy/Sell */}
      {chartData.length > 0 && (
        <div className="mt-10 w-full max-w-6xl bg-white p-6 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">Signal Chart</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="price" stroke="#0ea5e9" name="Price" />
              <Scatter data={buySignals} fill="#22c55e" name="Buy" shape="triangle" />
              <Scatter data={sellSignals} fill="#ef4444" name="Sell" shape="cross" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Equity Curve */}
      {equityCurve && (
        <div className="mt-10 w-full max-w-6xl bg-white p-6 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">Equity Curve</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={equityCurve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index">
                <Label value="Trade #" position="insideBottom" />
              </XAxis>
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="capital"
                stroke="#10b981"
                strokeWidth={2}
                name="Equity Value"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
