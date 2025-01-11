"use client";

import React, { useEffect, useState } from "react";
import axios from "axios";

export function StockMarketNews() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch news from a stock market news API
    const fetchNews = async () => {
      try {
        const response = await axios.get(
          `https://newsapi.org/v2/everything?q=stock+market&apiKey=${process.env.NEXT_PUBLIC_Stock_News}`
        );
        setNews(response.data.articles.slice(0, 10)); // Limiting to 10 articles
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch news");
        setLoading(false);
      }
    };

    fetchNews();
  }, []);

  if (loading) return <div>Loading stock market news...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="p-4 bg-gray-100 rounded-md">
      <h2 className="text-xl font-semibold mb-4">Stock Market News & Updates</h2>
      <ul className="space-y-4">
        {news.map((article, index) => (
          <li key={index} className="p-4 bg-white shadow-sm rounded-md">
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 font-bold hover:underline"
            >
              {article.title}
            </a>
            <p className="text-sm text-gray-600 mt-2">{article.source.name}</p>
            <p className="text-sm text-gray-500">
              Published on: {new Date(article.publishedAt).toLocaleString()}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
