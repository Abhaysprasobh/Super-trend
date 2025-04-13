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
} from "recharts";

// Custom dot for Buy/Sell signals
  const CustomSignalDot = (props) => {
    const { cx, cy, payload } = props;
    if (payload.up1 === 1) {
      return (
        <text
          x={cx}
          y={cy}
          dy={-10}
          textAnchor="middle"
          fill="green"
          fontSize={20}
        >
          ▲
        </text>
      );
    } else if (payload.up1 === 0) {
      return (
        <text x={cx} y={cy} dy={10} textAnchor="middle" fill="red" fontSize={20}>
          ▼
        </text>
      );
    }
    return null;
  };

export default function AdaptiveSupertrendPage() {
  const [formData, setFormData] = useState({
    ticker: "AAPL",
    atr_len: 14,
    training_data_period: 100,

    high_multiplier: 4.0,
    mid_multiplier: 3.0,
    low_multiplier: 2.0,
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
      const raw = await fetchAdaptiveSupertrend(formData);
      const response = typeof raw === "string" ? JSON.parse(raw) : raw;
      console.log("Response:", response);
      setResult(response);
    } catch (err) {
      console.error(err);
      setError("Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const chartData = result?.data || [];

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

      {/* Chart */}
      {chartData.length > 0 && (
        <div className="mt-10 w-full max-w-6xl bg-white p-6 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">
            Adaptive Supertrend Chart – {formData.ticker.toUpperCase()}
          </h2>
          <ResponsiveContainer width="100%" height={600}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Date" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Legend />

              {/* Close Price */}
              <Line
                type="monotone"
                dataKey="Close"
                stroke="#0ea5e9"
                name="Close Price"
                dot={<CustomSignalDot />}
              />

              {/* Adaptive Supertrend */}
              <Line
                type="monotone"
                dataKey="ADAPT_SUPERT"
                stroke="#f97316"
                name="Adaptive Supertrend"
                dot={false}
              />

              {/* Buy Signal (Legend only, green) */}
              <Line
                type="monotone"
                dataKey="Buy"
                stroke="#22c55e"
                name="Uptrend"
                dot={false}
                // hide={true}
              />

              {/* Sell Signal (Legend only, red) */}
              <Line
                type="monotone"
                dataKey="Sell"
                stroke="#ef4444"
                name="Down Trend"
                dot={false}
                // hide={true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
