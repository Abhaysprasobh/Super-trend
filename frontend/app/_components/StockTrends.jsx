"use client";

import React, { useState, useEffect } from "react";
import { TrendingUp } from "lucide-react";

const API_URL = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT,GOOGL,AAPL,TSLA&apikey=${process.env.NEXT_PUBLIC_STOCK_API_KEY}`;

export default function StockTrends() {
  const [risingStocks, setRisingStocks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStockData() {
      try {
        const stockSymbols = ["MSFT", "GOOGL", "AAPL", "TSLA"];
        const risingStockData = [];

        for (const symbol of stockSymbols) {
          const response = await fetch(
            `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=${symbol}&apikey=${process.env.NEXT_PUBLIC_STOCK_API_KEY}`
          );
          const data = await response.json();

          if (data["Time Series (Daily)"]) {
            const timeSeries = data["Time Series (Daily)"];
            const dates = Object.keys(timeSeries).slice(0, 2);
            const todayPrice = parseFloat(timeSeries[dates[0]]["4. close"]);
            const previousPrice = parseFloat(timeSeries[dates[1]]["4. close"]);

            const priceChange = ((todayPrice - previousPrice) / previousPrice) * 100;

            if (priceChange > 0) {
              risingStockData.push({
                symbol,
                todayPrice: todayPrice.toFixed(2),
                change: priceChange.toFixed(2),
              });
            }
          }
        }

        setRisingStocks(risingStockData);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching stock data:", error);
      }
    }

    fetchStockData();
  }, []);

  return (
    <div className="p-6 bg-gray-100 rounded shadow-md max-w-md mx-auto">
      <h2 className="text-2xl font-bold mb-4">Rising Stocks</h2>
      {loading ? (
        <p>Loading stock market trends...</p>
      ) : (
        <ul className="space-y-4">
          {risingStocks.length > 0 ? (
            risingStocks.map((stock) => (
              <li key={stock.symbol} className="flex justify-between">
                <div>
                  <span className="font-medium text-lg">{stock.symbol}</span>
                </div>
                <div className="text-green-600 flex items-center">
                  <TrendingUp className="h-5 w-5 mr-1" />
                  ${stock.todayPrice} (+{stock.change}%)
                </div>
              </li>
            ))
          ) : (
            <p>No rising stocks at the moment.</p>
          )}
        </ul>
      )}
    </div>
  );
}
