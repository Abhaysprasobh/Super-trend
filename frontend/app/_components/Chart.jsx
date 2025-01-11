"use client";
import * as React from "react";
import { Area, AreaChart, CartesianGrid, XAxis } from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const chartData = [
  { date: "2024-04-01", investment: 222, deposit: 150 },
  { date: "2024-04-02", investment: 97, deposit: 180 },
  { date: "2024-04-03", investment: 167, deposit: 120 },
  { date: "2024-04-04", investment: 242, deposit: 260 },
  { date: "2024-04-05", investment: 373, deposit: 290 },
  { date: "2024-04-06", investment: 301, deposit: 340 },
  { date: "2024-04-07", investment: 245, deposit: 180 },
  { date: "2024-04-08", investment: 409, deposit: 320 },
  { date: "2024-04-09", investment: 59, deposit: 110 },
  { date: "2024-04-10", investment: 261, deposit: 190 },
  { date: "2024-04-11", investment: 327, deposit: 350 },
  { date: "2024-04-12", investment: 292, deposit: 210 },
  { date: "2024-04-13", investment: 342, deposit: 380 },
  { date: "2024-04-14", investment: 137, deposit: 220 },
  { date: "2024-04-15", investment: 120, deposit: 170 },
  { date: "2024-04-16", investment: 138, deposit: 190 },
  { date: "2024-04-17", investment: 446, deposit: 360 },
  { date: "2024-04-18", investment: 364, deposit: 410 },
  { date: "2024-04-19", investment: 243, deposit: 180 },
  { date: "2024-04-20", investment: 89, deposit: 150 },
  { date: "2024-04-21", investment: 137, deposit: 200 },
  { date: "2024-04-22", investment: 224, deposit: 170 },
  { date: "2024-04-23", investment: 138, deposit: 230 },
  { date: "2024-04-24", investment: 387, deposit: 290 },
  { date: "2024-04-25", investment: 215, deposit: 250 },
  { date: "2024-04-26", investment: 75, deposit: 130 },
  { date: "2024-04-27", investment: 383, deposit: 420 },
  { date: "2024-04-28", investment: 122, deposit: 180 },
  { date: "2024-04-29", investment: 315, deposit: 240 },
  { date: "2024-04-30", investment: 454, deposit: 380 },
  { date: "2024-05-01", investment: 165, deposit: 220 },
  { date: "2024-05-02", investment: 293, deposit: 310 },
  { date: "2024-05-03", investment: 247, deposit: 190 },
  { date: "2024-05-04", investment: 385, deposit: 420 },
  { date: "2024-05-05", investment: 481, deposit: 390 },
  { date: "2024-05-06", investment: 498, deposit: 520 },
  { date: "2024-05-07", investment: 388, deposit: 300 },
  { date: "2024-05-08", investment: 149, deposit: 210 },
  { date: "2024-05-09", investment: 227, deposit: 180 },
  { date: "2024-05-10", investment: 293, deposit: 330 },
  { date: "2024-05-11", investment: 335, deposit: 270 },
  { date: "2024-05-12", investment: 197, deposit: 240 },
  { date: "2024-05-13", investment: 197, deposit: 160 },
  { date: "2024-05-14", investment: 448, deposit: 490 },
  { date: "2024-05-15", investment: 473, deposit: 380 },
  { date: "2024-05-16", investment: 338, deposit: 400 },
  { date: "2024-05-17", investment: 499, deposit: 420 },
  { date: "2024-05-18", investment: 315, deposit: 350 },
  { date: "2024-05-19", investment: 235, deposit: 180 },
  { date: "2024-05-20", investment: 177, deposit: 230 },
  { date: "2024-05-21", investment: 82, deposit: 140 },
  { date: "2024-05-22", investment: 81, deposit: 120 },
  { date: "2024-05-23", investment: 252, deposit: 290 },
  { date: "2024-05-24", investment: 294, deposit: 220 },
  { date: "2024-05-25", investment: 201, deposit: 250 },
  { date: "2024-05-26", investment: 213, deposit: 170 },
  { date: "2024-05-27", investment: 420, deposit: 460 },
  { date: "2024-05-28", investment: 233, deposit: 190 },
  { date: "2024-05-29", investment: 78, deposit: 130 },
  { date: "2024-05-30", investment: 340, deposit: 280 },
  { date: "2024-05-31", investment: 178, deposit: 230 },
  { date: "2024-06-01", investment: 178, deposit: 200 },
  { date: "2024-06-02", investment: 470, deposit: 410 },
  { date: "2024-06-03", investment: 103, deposit: 160 },
  { date: "2024-06-04", investment: 439, deposit: 380 },
  { date: "2024-06-05", investment: 88, deposit: 140 },
  { date: "2024-06-06", investment: 294, deposit: 250 },
  { date: "2024-06-07", investment: 323, deposit: 370 },
  { date: "2024-06-08", investment: 385, deposit: 320 },
  { date: "2024-06-09", investment: 438, deposit: 480 },
  { date: "2024-06-10", investment: 155, deposit: 200 },
  { date: "2024-06-11", investment: 92, deposit: 150 },
  { date: "2024-06-12", investment: 492, deposit: 420 },
  { date: "2024-06-13", investment: 81, deposit: 130 },
  { date: "2024-06-14", investment: 426, deposit: 380 },
  { date: "2024-06-15", investment: 307, deposit: 350 },
  { date: "2024-06-16", investment: 371, deposit: 310 },
  { date: "2024-06-17", investment: 475, deposit: 520 },
  { date: "2024-06-18", investment: 107, deposit: 170 },
  { date: "2024-06-19", investment: 341, deposit: 290 },
  { date: "2024-06-20", investment: 408, deposit: 450 },
  { date: "2024-06-21", investment: 169, deposit: 210 },
  { date: "2024-06-22", investment: 317, deposit: 270 },
  { date: "2024-06-23", investment: 480, deposit: 530 },
  { date: "2024-06-24", investment: 132, deposit: 180 },
  { date: "2024-06-25", investment: 141, deposit: 190 },
  { date: "2024-06-26", investment: 434, deposit: 380 },
  { date: "2024-06-27", investment: 448, deposit: 490 },
  { date: "2024-06-28", investment: 149, deposit: 200 },
  { date: "2024-06-29", investment: 103, deposit: 160 },
  { date: "2024-06-30", investment: 446, deposit: 400 },
];

const chartConfig = {
  visitors: {
    label: "Visitors",
  },
  investment: {
    label: "Desktop",
    color: "hsl(var(--chart-1))",
  },
  deposit: {
    label: "Mobile",
    color: "hsl(var(--chart-2))",
  },
};

export function Chart() {
  const [timeRange, setTimeRange] = React.useState("90d");
  const filteredData = chartData.filter((item) => {
    const date = new Date(item.date);
    const referenceDate = new Date("2024-06-30");
    let daysToSubtract = 90;
    if (timeRange === "30d") {
      daysToSubtract = 30;
    } else if (timeRange === "7d") {
      daysToSubtract = 7;
    }
    const startDate = new Date(referenceDate);
    startDate.setDate(startDate.getDate() - daysToSubtract);
    return date >= startDate;
  });

  return (
    <Card>
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1 text-center sm:text-left">
          <CardTitle>Area Chart - Interactive</CardTitle>
          <CardDescription>
            Showing total visitors for the last 3 months
          </CardDescription>
        </div>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[160px] rounded-lg sm:ml-auto" aria-label="Select a value">
            <SelectValue placeholder="Last 3 months" />
          </SelectTrigger>
          <SelectContent className="rounded-xl">
            <SelectItem value="90d" className="rounded-lg">
              Last 3 months
            </SelectItem>
            <SelectItem value="30d" className="rounded-lg">
              Last 30 days
            </SelectItem>
            <SelectItem value="7d" className="rounded-lg">
              Last 7 days
            </SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer config={chartConfig} className="aspect-auto h-[250px] w-full">
          <AreaChart data={filteredData}>
            <defs>
              <linearGradient id="fillDesktop" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="red" stopOpacity={0.1} />
                <stop offset="95%" stopColor="red" stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="fillMobile" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="green" stopOpacity={0.1} />
                <stop offset="95%" stopColor="green" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="investment"
              stroke="red"
              fill="url(#fillDesktop)"
            />
            <Area
              type="monotone"
              dataKey="deposit"
              stroke="green"
              fill="url(#fillMobile)"
            />
            <CartesianGrid opacity={0.1} />
            <XAxis dataKey="date" />
          </AreaChart>
          <ChartTooltip>
            <ChartTooltipContent label="Date" />
            <ChartTooltipContent
              label="investment"
              value="investment"
              color="green"
            />
            <ChartTooltipContent
              label="deposit"
              value="deposit"
              color="red"
            />
          </ChartTooltip>
          <ChartLegend>
            <ChartLegendContent label={chartConfig.investment.label} color={chartConfig.investment.color} />
            <ChartLegendContent label={chartConfig.deposit.label} color={chartConfig.deposit.color} />
          </ChartLegend>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
