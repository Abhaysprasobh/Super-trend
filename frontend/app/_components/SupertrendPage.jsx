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
  Scatter,
} from "recharts";

export default function SupertrendPage() {
  const [formData, setFormData] = useState({
    ticker: "AAPL",
    interval: "1d",
    days: 700,
    length: 7,
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
      const response = await fetchSupertrend(formData);
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
  const buySignals = chartData.filter((d) => d.signal === "buy");
  const sellSignals = chartData.filter((d) => d.signal === "sell");

  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-6">Standard Supertrend</h1>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded-2xl shadow-md w-full max-w-3xl grid grid-cols-1 md:grid-cols-2 gap-4"
      >
        {Object.keys(formData).map((key) => (
          <div key={key} className="flex flex-col">
            <label htmlFor={key} className="text-sm font-semibold capitalize">
              {key}
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
      {loading && !error && (
        <div className="mt-10 text-blue-600 font-semibold">
          Fetching chart data...
        </div>
      )}

      {!loading && chartData.length === 0 && result && (
        <p className="text-gray-500 mt-4">
          No chart data available for this configuration.
        </p>
      )}

      {chartData.length > 0 && (
        <div className="mt-10 w-full max-w-6xl bg-white p-6 rounded-xl shadow">
          <h2 className="text-xl font-semibold mb-4">Supertrend Chart</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="close"
                stroke="#3b82f6"
                name="Close Price"
              />
              <Line
                type="monotone"
                dataKey="supertrend"
                stroke="#f97316"
                name="Supertrend"
              />
              <Scatter
                data={buySignals}
                fill="#16a34a"
                name="Buy Signal"
                shape="triangle"
              />
              <Scatter
                data={sellSignals}
                fill="#dc2626"
                name="Sell Signal"
                shape="cross"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
