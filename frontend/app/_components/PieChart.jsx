"use client";

import React, { useEffect, useState } from "react";
import { TrendingUp } from "lucide-react";
import { ResponsiveContainer, PieChart, Pie, Label } from "recharts";
import { fetchStockData } from "../_utils/GlobalApi"; 
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function CustomPieChart() {
  const [ticker, setTicker] = useState("AAPL");
  const [chartData, setChartData] = useState([]);
  const [avgClosePrice, setAvgClosePrice] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userInput, setUserInput] = useState("");
  const [stockInfo, setStockInfo] = useState({
    currentPrice: 0,
    dayHigh: 0,
    dayLow: 0,
  });


const loadStockData = async (selectedTicker) => {
  try {
    setLoading(true);
    setError(null);

    const stockData = await fetchStockData(selectedTicker);
    console.log("Stock Data Response:", stockData);

    if (!stockData || typeof stockData !== "object") {
      throw new Error("Invalid response from server.");
    }

    if (!stockData.history || !Array.isArray(stockData.history)) {
      throw new Error("Stock history data is missing or incorrect.");
    }

    setStockInfo({
      currentPrice: stockData.info?.currentPrice ?? 0,
      dayHigh: stockData.info?.dayHigh ?? 0,
      dayLow: stockData.info?.dayLow ?? 0,
    });

    const history = stockData.history
      .map((item) => ({
        date: new Date(item.Date).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        }),
        close: item.Close ?? 0,
      }))
      .filter((item) => !isNaN(item.close) && item.close > 0);

    const totalClose = history.reduce((sum, item) => sum + item.close, 0);
    const avgClose = history.length > 0 ? totalClose / history.length : 0;

    setChartData(history);
    setAvgClosePrice(avgClose);
  } catch (err) {
    setError(err.message || "Failed to fetch stock data.");
    console.error("Stock data fetch error:", err);
  } finally {
    setLoading(false);
  }
};

  // useEffect(() => {
  //   loadStockData(ticker);
  // }, [ticker]);

  const handleSearch = () => {
    if (userInput.trim()) {
      setTicker(userInput.toUpperCase());loadStockData(ticker);
    }
  };

  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle>Stock Chart</CardTitle>
        <CardDescription>Enter a stock ticker to view trends</CardDescription>
      </CardHeader>

      {/* User Input */}
      <CardContent className="flex flex-row gap-2 pb-4">
        <Input
          type="text"
          placeholder="Enter ticker (e.g., AAPL)"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
        />
        <Button onClick={handleSearch}>Search</Button>
      </CardContent>

      {/* Loading & Error Handling */}
      {loading ? (
        <div className="text-gray-500">Loading stock data...</div>
      ) : error ? (
        <div className="text-red-500">Error: {error}</div>
      ) : (
        <>
          <CardContent className="flex-1 pb-0">
            <div className="mx-auto aspect-square max-h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    dataKey="close"
                    nameKey="date"
                    innerRadius={60}
                    outerRadius={80}
                    strokeWidth={5}
                  >
                    <Label
                      content={({ viewBox }) => {
                        if (viewBox?.cx && viewBox?.cy) {
                          return (
                            <text
                              x={viewBox.cx}
                              y={viewBox.cy}
                              textAnchor="middle"
                              dominantBaseline="middle"
                            >
                              <tspan className="fill-foreground text-3xl font-bold">
                                {avgClosePrice.toFixed(2)}
                              </tspan>
                              <tspan
                                x={viewBox.cx}
                                y={viewBox.cy + 24}
                                className="fill-muted-foreground text-sm"
                              >
                                Avg Close Price
                              </tspan>
                            </text>
                          );
                        }
                        return null;
                      }}
                    />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <CardContent className="text-center">
              <div className="text-xl font-semibold">
                {ticker} - ${stockInfo.currentPrice.toFixed(2)}
              </div>
              <div className="text-sm text-muted-foreground">
                High: ${stockInfo.dayHigh.toFixed(2)} | Low: $
                {stockInfo.dayLow.toFixed(2)}
              </div>
            </CardContent>
          </CardContent>

          <CardFooter className="flex-col gap-2 text-sm">
            {/* <div className="flex items-center gap-2 font-medium leading-none">
              Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
            </div> */}
            {/* <div className="leading-none text-muted-foreground">
              Showing closing prices for {ticker}
            </div> */}
          </CardFooter>
        </>
      )}
    </Card>
  );
}
