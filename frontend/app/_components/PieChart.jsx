"use client";

import React, { useEffect, useState, useMemo } from "react";
import { TrendingUp } from "lucide-react";
import { ResponsiveContainer, PieChart as RePieChart, Pie, Label } from "recharts";
import { fetchStockData } from "../_utils/GlobalApi"; 

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

export function CustomPieChart({ ticker = "AAPL" }) {
  const [chartData, setChartData] = useState([]);
  const [totalVisitors, setTotalVisitors] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadStockData = async () => {
      try {
        setLoading(true);
        setError(null);

        const stockData = await fetchStockData(ticker);
        const history = stockData.history.map((item) => ({
          date: item.date,
          visitors: item.close, // Assuming "Close" price as visitors for visualization
          fill: "hsl(var(--chart-1))",
        }));

        setChartData(history);
        setTotalVisitors(history.reduce((acc, curr) => acc + curr.visitors, 0));
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    loadStockData();
  }, [ticker]);

  if (loading) return <div>Loading stock data...</div>;
  if (error) return <div className="text-red-500">Error: {error}</div>;

  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle>{ticker} Stock Chart</CardTitle>
        <CardDescription>Last 1 Month Closing Prices</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer className="mx-auto aspect-square max-h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <RePieChart>
              <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
              <Pie
                data={chartData}
                dataKey="visitors"
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
                            {totalVisitors.toFixed(2)}
                          </tspan>
                          <tspan x={viewBox.cx} y={viewBox.cy + 24} className="fill-muted-foreground">
                            Avg Close Price
                          </tspan>
                        </text>
                      );
                    }
                    return null;
                  }}
                />
              </Pie>
            </RePieChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col gap-2 text-sm">
        <div className="flex items-center gap-2 font-medium leading-none">
          Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
        </div>
        <div className="leading-none text-muted-foreground">
          Showing closing prices for the last 1 month
        </div>
      </CardFooter>
    </Card>
  );
}
