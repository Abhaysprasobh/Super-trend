"use client";

import React, { useState, useEffect } from "react";
import { fetchIndicatorComparison } from "../_utils/GlobalApi"; // API request
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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
    <Card className="p-6 border rounded-lg shadow-md bg-white">
      <CardHeader>
        <CardTitle>Stock Indicator Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Symbol Input */}
        <div className="flex gap-4">
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter stock symbol (e.g., AAPL)"
            className="p-2 border rounded w-1/2"
          />
          
          {/* Duration Dropdown */}
          <select
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
            className="p-2 border rounded w-1/2"
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
            className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Get Data
          </button>
        </div>

        {/* Status Messages */}
        {loading && <p>Loading indicator data...</p>}
        {error && <p className="text-red-500">Error: {error}</p>}

        {/* Data Display */}
        {data && (
          <div className="mt-6">
            <h3 className="font-medium text-gray-700">Backtest Performance</h3>
            <p><strong>Win Rate:</strong> {data.backtest?.winRate}%</p>
            <p><strong>Annual Return:</strong> {data.backtest?.annualReturn}%</p>

            <h3 className="mt-4 font-medium text-gray-700">Signals</h3>
            {data.signals?.length ? (
              <ul className="list-disc list-inside">
                {data.signals.map((signal, index) => (
                  <li key={index}>
                    {signal.date}: {signal.type} at ${signal.price.toFixed(2)}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No signals available</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default IndicatorComparison;
