"use client";

import React, { useState } from "react";
import { fetchIndicatorComparison } from "../_utils/GlobalApi";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

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
      const days = durationOptions[duration];
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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter Stock Symbol (e.g., AAPL)"
            className="w-full p-3 border rounded-lg text-gray-700 focus:ring focus:ring-blue-300 outline-none"
          />

          <select
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
            className="w-full p-3 border rounded-lg text-gray-700 focus:ring focus:ring-blue-300 outline-none"
          >
            {Object.keys(durationOptions).map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>

          <button
            onClick={handleFetchData}
            className="w-full p-3 bg-gradient-to-r from-blue-500 to-cyan-600 text-white font-semibold rounded-lg hover:scale-105 transition-transform duration-200 flex items-center justify-center"
          >
            {loading ? <Loader2 className="animate-spin h-5 w-5" /> : "Get Data"}
          </button>
        </div>

        {error && <p className="text-red-500 mt-4 text-sm">{error}</p>}
        {loading && <p className="text-gray-500 mt-4 text-sm">Fetching data...</p>}

        {data && (
          <div className="mt-6 space-y-6">
            <div className="p-4 border rounded-lg bg-gray-50 shadow-sm">
              <h3 className="text-lg font-medium text-gray-800">📊 Performance Summary</h3>
              {Object.entries(data.performance || {}).map(([strategy, stats]) => (
                <div key={strategy} className="text-sm text-gray-600 mt-2">
                  <strong>{strategy}</strong>
                  <p>Total Return: {stats?.total_return || "N/A"}</p>
                  <p>Annualized Return: {stats?.annualized_return || "N/A"}</p>
                  <p>Sharpe Ratio: {stats?.sharpe_ratio || "N/A"}</p>
                  <p>Win Rate: {stats?.win_rate || "N/A"}</p>
                </div>
              ))}
            </div>

            <div className="p-4 border rounded-lg bg-gray-50 shadow-sm">
              <h3 className="text-lg font-medium text-gray-800">📢 Trading Signals</h3>
              {Object.entries(data.signals || {}).map(([strategy, signalData]) => (
                <div key={strategy} className="mt-2">
                  <strong>{strategy}</strong>
                  {signalData.dates?.length ? (
                    <ul className="mt-2 space-y-2">
                      {signalData.dates.map((date, index) => (
                        <li key={index} className="flex justify-between items-center text-sm text-gray-700 border-b pb-2">
                          <span>{date}</span>
                          <span className={`font-semibold ${signalData.signals[index] === 1 ? "text-green-500" : "text-red-500"}`}>
                            {signalData.signals[index] === 1 ? "BUY" : "SELL"} at ${signalData.prices[index]?.toFixed(2) || "N/A"}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-600 text-sm mt-1">No signals available</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default IndicatorComparison;