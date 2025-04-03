"use client";

import React, { useState } from "react";
import { fetchIndicatorComparison } from "../_utils/GlobalApi"; // API request
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, TrendingUp } from "lucide-react";

const durationOptions = {
  "1 Month": 30,
  "3 Months": 90,
  "6 Months": 180,
  "1 Year": 365,
  "2 Years": 730,
};

const IndicatorComparison = () => {
  const [symbol, setSymbol] = useState("AAPL");
  const [duration, setDuration] = useState("1 Month");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      setData(null);
      const days = durationOptions[duration]; // Convert duration to days
      const result = await fetchIndicatorComparison(symbol, "1d", days);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="p-6 border rounded-xl shadow-xl bg-white max-w-3xl mx-auto">
      <CardHeader className="pb-4">
        <CardTitle className="text-xl font-semibold text-gray-900 flex items-center gap-2">
          📈 Stock Indicator Comparison
        </CardTitle>
      </CardHeader>

      <CardContent>
        {/* Input Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Stock Symbol Input */}
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter Stock Symbol (e.g., AAPL)"
            className="w-full p-3 border rounded-lg text-gray-700 focus:ring focus:ring-blue-300 outline-none"
          />

          {/* Duration Dropdown */}
          <select
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
            className="w-full p-3 border rounded-lg text-gray-700 focus:ring focus:ring-blue-300 outline-none"
          >
            {Object.keys(durationOptions).map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>

          {/* Fetch Button */}
          <button
            onClick={handleFetchData}
            className="w-full p-3 bg-gradient-to-r from-blue-500 to-cyan-600 text-white font-semibold rounded-lg hover:scale-105 transition-transform duration-200 flex items-center justify-center"
          >
            {loading ? <Loader2 className="animate-spin h-5 w-5" /> : "Get Data"}
          </button>
        </div>

        {/* Status Messages */}
        {error && <p className="text-red-500 mt-4 text-sm">{error}</p>}
        {loading && <p className="text-gray-500 mt-4 text-sm">Fetching data...</p>}

        {/* Data Display */}
        {data && (
          <div className="mt-6 space-y-6">
            {/* Backtest Performance */}
            <div className="p-4 border rounded-lg bg-gray-50 shadow-sm">
              <h3 className="text-lg font-medium text-gray-800">📊 Backtest Performance</h3>
              <p className="text-sm text-gray-600 mt-1">
                <strong>Win Rate:</strong> {data.backtest?.winRate}%
              </p>
              <p className="text-sm text-gray-600">
                <strong>Annual Return:</strong> {data.backtest?.annualReturn}%
              </p>
            </div>

            {/* Trading Signals */}
            <div className="p-4 border rounded-lg bg-gray-50 shadow-sm">
              <h3 className="text-lg font-medium text-gray-800">📢 Trading Signals</h3>
              {data.signals?.length ? (
                <ul className="mt-2 space-y-2">
                  {data.signals.map((signal, index) => (
                    <li
                      key={index}
                      className="flex justify-between items-center text-sm text-gray-700 border-b pb-2"
                    >
                      <span>{signal.date}</span>
                      <span className={`font-semibold ${signal.type === "BUY" ? "text-green-500" : "text-red-500"}`}>
                        {signal.type} at ${signal.price.toFixed(2)}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-600 text-sm mt-1">No signals available</p>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default IndicatorComparison;
