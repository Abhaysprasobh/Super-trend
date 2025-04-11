"use client";

import React, { useState } from "react";
import { fetchIndicatorComparison } from "../_utils/GlobalApi";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Scatter,
  Legend,
  LabelList,
} from "recharts";

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
        length :14,
        days,
      };

      const result = await fetchIndicatorComparison(payload);

      const convertToSignals = (rawData) =>
        rawData.map((entry) => ({
          date: entry.date,
          close: entry.close,
          trend_value: entry.trend_value,
          buy_signal: entry.buy_signal === 1 ? entry.close : null,
          sell_signal: entry.sell_signal === 1 ? entry.close : null,
        }));

      setData({
        ...result,
        adaptive_chart: convertToSignals(result.adapt || []),
        standard_chart: convertToSignals(result.super || []),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="p-6 border rounded-xl shadow-xl bg-white max-w-6xl mx-auto">
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
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter Stock Symbol (e.g., AAPL)"
            className="w-full p-3 border rounded-lg"
          />
          <select
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
            className="w-full p-3 border rounded-lg"
          >
            {Object.keys(durationOptions).map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
          <button
            onClick={handleFetchData}
            className="w-full p-3 bg-blue-600 text-white rounded-lg hover:opacity-90"
          >
            {loading ? (
              <Loader2 className="animate-spin h-5 w-5" />
            ) : (
              "Get Data"
            )}
          </button>
        </div>

        {error && <p className="text-red-500 mt-4 text-sm">{error}</p>}

        {data && (
          <div className="mt-6 space-y-8">
            {/* Performance Table */}
            <div className="p-4 border rounded-lg bg-gray-50">
              <h3 className="text-lg font-semibold mb-3">
                📐 Strategy Performance
              </h3>
              <table className="min-w-full text-sm text-left text-gray-700 border">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="p-3 border">Strategy</th>
                    <th className="p-3 border">Final Capital</th>
                    <th className="p-3 border">Total Trades</th>
                    <th className="p-3 border">Profitable Trades</th>
                    <th className="p-3 border">Total Return (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {["adaptive", "Supertrend"].map((key) => {
                    const perf = data.performance[key];
                    return (
                      <tr key={key} className="border-t">
                        <td className="p-3 border font-medium capitalize">
                          {key}
                        </td>
                        <td className="p-3 border">
                          ₹{perf.final_capital?.toFixed(2)}
                        </td>
                        <td className="p-3 border">{perf.total_trades}</td>
                        <td className="p-3 border">{perf.profitable_trades}</td>
                        <td className="p-3 border">
                          {perf.total_return?.toFixed(2)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Chart Comparison */}
            {["adaptive", "standard"].map((type) => {
              const chartData = data[`${type}_chart`];
              const title =
                type === "adaptive"
                  ? "🤖 Adaptive Strategy"
                  : "📐 Standard Strategy";

              return (
                <div key={type} className="p-4 border rounded-lg bg-gray-50">
                  <h3 className="text-lg font-semibold mb-4">{title}</h3>
                  {chartData?.length > 0 ? (
                    <ResponsiveContainer width="100%" height={350}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />

                        {/* Close and Trend */}
                        <Line
                          type="monotone"
                          dataKey="close"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={false}
                          name="Close"
                        />
                        <Line
                          type="monotone"
                          dataKey="trend_value"
                          stroke="#10b981"
                          strokeWidth={2}
                          dot={false}
                          name="Trend"
                        />

                        {/* Buy Signal */}
                        <Scatter
                          data={chartData}
                          dataKey="buy_signal"
                          fill="#22c55e"
                          shape="triangle"
                          name="Buy Signal"
                        >
                          <LabelList
                            dataKey="buy_signal"
                            position="top"
                            fill="#22c55e"
                            fontSize={12}
                          />
                        </Scatter>

                        {/* Sell Signal */}
                        <Scatter
                          data={chartData}
                          dataKey="sell_signal"
                          fill="#ef4444"
                          shape="cross"
                          name="Sell Signal"
                        >
                          <LabelList
                            dataKey="sell_signal"
                            position="bottom"
                            fill="#ef4444"
                            fontSize={12}
                          />
                        </Scatter>
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-sm text-gray-500">
                      No chart data available.
                    </p>
                  )}
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
