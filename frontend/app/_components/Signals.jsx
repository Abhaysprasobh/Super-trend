"use client";

import React, { useState } from "react";
import { fetchIndicatorComparison } from "../_utils/GlobalApi";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

const durationOptions = {
  "6 Months": 180,
  "1 Year": 365,
  "2 Years": 700,
};

const IndicatorComparison = () => {
  const [symbol, setSymbol] = useState("AAPL");
  const [duration, setDuration] = useState("1 Year");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFetchData = async () => {
    const days = durationOptions[duration] ?? 365;
    if (!symbol) {
      setError("Please enter a stock symbol.");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setData(null);

      const payload = {
        symbol,
        high_vol_multiplier: 3,
        mid_vol_multiplier: 2,
        low_vol_multiplier: 1,
        days,
      };

      const result = await fetchIndicatorComparison(payload);
      console.log(result);
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
          📈 Adaptive vs Standard Strategy Comparison
        </CardTitle>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol((e?.target?.value || "").toUpperCase())}
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
      <h3 className="text-lg font-medium text-gray-800">📊 Capital Comparison</h3>
      <p className="text-sm text-gray-600 mt-2">
        🧠 <strong>Better Strategy:</strong>{" "}
        <span className="uppercase font-semibold text-blue-600">
          {data.better_strategy}
        </span>
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4 text-sm text-gray-700">
        <div>
          <strong>Adaptive Final Capital:</strong>{" "}
          ₹{data.adaptive?.final_capital?.toFixed(2) ?? "N/A"}
        </div>
        <div>
          <strong>Standard Final Capital:</strong>{" "}
          ₹{data.standard?.final_capital?.toFixed(2) ?? "N/A"}
        </div>
      </div>
    </div>

    {["adaptive", "standard"].map((type) => {
      const strategy = data[type];
      return (
        <div
          key={type}
          className="p-4 border rounded-lg bg-gray-50 shadow-sm"
        >
          <h3 className="text-lg font-medium text-gray-800">
            {type === "adaptive" ? "🤖 Adaptive Strategy" : "📐 Standard Strategy"}
          </h3>

          <div className="mt-3">
            <p className="font-semibold text-gray-700 mb-1">Buy Signals:</p>
            <ul className="text-sm space-y-1 text-green-700">
              {strategy?.buy_signals?.length > 0 ? (
                strategy.buy_signals.map((sig, i) => (
                  <li key={i}>
                    🟢 {sig.date} — ₹{sig.price.toFixed(2)}
                  </li>
                ))
              ) : (
                <li>No buy signals</li>
              )}
            </ul>
          </div>

          <div className="mt-3">
            <p className="font-semibold text-gray-700 mb-1">Sell Signals:</p>
            <ul className="text-sm space-y-1 text-red-700">
              {strategy?.sell_signals?.length > 0 ? (
                strategy.sell_signals.map((sig, i) => (
                  <li key={i}>
                    🔴 {sig.date} — ₹{sig.price.toFixed(2)}
                  </li>
                ))
              ) : (
                <li>No sell signals</li>
              )}
            </ul>
          </div>
        </div>
      );
    })}
  </div>
)}
      </CardContent>
    </Card>
  );
};

export default IndicatorComparison;
