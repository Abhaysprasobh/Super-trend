"use client";
import React, { useState } from 'react'
import Image from 'next/image'
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import Link from 'next/link';


function Header() {
    const [isLogin, setIsLogin] = useState(false);

    // Move localStorage logic into useEffect to avoid hydration errors
    React.useEffect(() => {
      const authToken = localStorage.getItem('authToken');
      if (authToken) {
        setIsLogin(true);
      }
    }, []);

    const handleLogout = React.useCallback(() => {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('authToken');
        setIsLogin(false);
      }
    }, []);
  return (
    <div>
        <header className="bg-white">
        <div className="mx-auto max-w-screen-xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 items-center justify-between">
            <div className="md:flex md:items-center md:gap-12">
                <a className="block text-teal-600" href="./">
                <span className="sr-only">Home</span>
                <Image src="./logo.svg" alt='logo' width={60} height={20}/>
                </a>
            </div>

            <div className="hidden md:block">
                <nav aria-label="Global">
                <ul className="flex items-center gap-6 text-sm">
                    <li>
                    <a className="text-gray-500 transition hover:text-gray-500/75" href="../StockTrends"> Market Trends </a>
                    </li>

                    <li>
                    <a className="text-gray-500 transition hover:text-gray-500/75" href="#"> Predictions & Insights </a>
                    </li>

                    <li>
                    <a className="text-gray-500 transition hover:text-gray-500/75" href="../StockNews"> News & Updates </a>
                    </li>

                    <li>
                    <a className="text-gray-500 transition hover:text-gray-500/75" href="#"> Trading Signals </a>
                    </li>
                </ul>
                </nav>
            </div>

            <div className="flex items-center gap-4">
                { !isLogin &&(
                <div className="sm:flex sm:gap-4">
                <a
                    className="rounded-md bg-teal-600 px-5 py-2.5 text-sm font-medium text-white shadow"
                    href="../Login"
                >
                    Login/Register
                </a>

                {/* <div className="hidden sm:flex">
                    <a
                    className="rounded-md bg-gray-100 px-5 py-2.5 text-sm font-medium text-teal-600"
                    href="../Register"
                    >
                    Register
                    </a>
                </div> */}
                </div>
                )}

                {isLogin && (
                    <Popover>
                    <PopoverTrigger>
                    <Image src="./profile.svg" alt='logo' width={40} height={20}/>
                    </PopoverTrigger>
                    <PopoverContent className="w-44">
                      <ul className="flex flex-col gap-2">
                        <Link href="../Portfolio" className="cursor-pointer hover:bg-slate-100 p-2 rounded-md">
                          My portfolio
                        </Link>
                        <li
                          className="cursor-pointer hover:bg-slate-100 p-2 rounded-md"
                          onClick={handleLogout}
                        >
                          Logout
                        </li>
                      </ul>
                    </PopoverContent>
                  </Popover>
                )}

                <div className="block md:hidden">
                <button className="rounded bg-gray-100 p-2 text-gray-600 transition hover:text-gray-600/75">
                    <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="size-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth="2"
                    >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                </button>
                </div>
            </div>
            </div>
        </div>
        </header>
    </div>
  )
}

export default Header
