import React from 'react';

export default function Hero() {
  return (
    <section className="flex flex-col items-center justify-center text-center py-24 px-6 bg-sky-950 text-gray-100 relative overflow-hidden">
      <div className="absolute inset-0 bg-grid-gray-300/[0.2] pointer-events-none"></div>
      <div className="max-w-3xl relative z-10">
        <h1 className="text-4xl font-extrabold md:text-6xl text-white drop-shadow-lg transform transition duration-500 hover:scale-105">
          Adaptive Super Trend Indicator
        </h1>
        <p className="text-gray-300 md:mt-6 md:text-lg font-medium">
          Gain precise market insights with our cutting-edge adaptive Super Trend indicator.
          Utilize real-time analytics, trend analysis, and dynamic alerts to make informed trading decisions.
        </p>
        <ul className="mt-6 text-gray-200 md:text-lg space-y-2 text-left md:text-center">
          <li className="flex items-center justify-center">
            ✅ Real-time trend analysis and adaptive strategies
          </li>
          <li className="flex items-center justify-center">
            ✅ Customizable alerts for major market movements
          </li>
          <li className="flex items-center justify-center">
            ✅ Seamless integration with trading platforms
          </li>
        </ul>
        <div className="mt-8 flex flex-col sm:flex-row sm:justify-center gap-4">
          <a
            href="#"
            className="inline-block rounded bg-green-600 px-12 py-3 text-lg font-medium text-white transition-transform transform hover:scale-105 hover:bg-green-700 focus:outline-none focus:ring focus:ring-green-400 shadow-lg"
          >
            Get Started
          </a>
          <a
            href="#"
            className="inline-block rounded border border-slate-600 px-12 py-3 text-lg font-medium text-green-300 transition-transform transform hover:scale-105 hover: focus:outline-none focus:ring focus:ring-green-400 shadow-lg"
          >
            Learn More
          </a>
        </div>
      </div>
    </section>
  );
}
