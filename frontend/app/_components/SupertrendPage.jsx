"use client";

import React, { useState } from "react";
import { fetchSupertrend } from "../_utils/GlobalApi";
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


const CustomSignalDot = ({ cx, cy, payload }) => {
  if (payload.is_buy) {
    return (
      <text x={cx -2} y={cy} dy={4} textAnchor="middle" fill="green" fontSize={10}>
        ▲
      </text>
    );
  } else if (payload.is_sell) {
    return (
      <text x={cx} y={cy} dy={4} textAnchor="middle" fill="red" fontSize={10}>
        ▼
      </text>
    );
  }
  return null;
};


export default function SupertrendPage() {
  const [formData, setFormData] = useState({
    ticker: "AAPL",
    days: 700,
    length: 14,
    multiplier: 3.0,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: ["days", "length", "multiplier"].includes(name)
        ? parseFloat(value)
        : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const raw = await fetchSupertrend(formData);
      const response = typeof raw === "string" ? JSON.parse(raw) : raw;
      console.log("Supertrend response:", response);
      setResult(response);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch Supertrend data.");
    } finally {
      setLoading(false);
    }
  };

  const chartData = result?.data || [];

  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-6">Standard Supertrend Strategy</h1>

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
              type={
                ["days", "length", "multiplier"].includes(key)
                  ? "number"
                  : "text"
              }
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

      {/* Error Message */}
      {error && <p className="text-red-500 mt-4">{error}</p>}

      {/* Chart */}
      {chartData.length > 0 && (
        <div className="mt-10 w-full max-w-6xl bg-white p-6 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">
            Standard Supertrend Chart – {formData.ticker.toUpperCase()}
          </h2>
          <ResponsiveContainer width="100%" height={600}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Legend />

              {/* Close Price */}
              <Line
                type="monotone"
                dataKey="close"
                stroke="#0ea5e9"
                name="Close Price"
                dot={<CustomSignalDot />}
              />

              {/* Supertrend */}
              <Line
                type="monotone"
                dataKey="supertrend"
                stroke="#f97316"
                name="Supertrend"
                dot={false}
              />

              {/* Buy Signal (Legend only, gray) */}
              <Line
                type="monotone"
                dataKey="is_buy"
                stroke="#22c55e"
                name="Uptrend"
                dot={false}
              />

              {/* Sell Signal (Legend only, gray) */}
              <Line
                type="monotone"
                stroke="#ef4444"
                name="Down Trend"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
