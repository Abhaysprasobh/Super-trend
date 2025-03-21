"use client";

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Image from 'next/image';

export default function Profile() {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const tok = localStorage.getItem("authToken");
    
    axios.get("http://127.0.0.1:5000/api/user", {
      headers: { "x-access-token": tok }
    })
    .then(response => {
      setProfile(response.data);
      setLoading(false);
    })
    .catch(error => {
      setError(error.message);
      setLoading(false);
    });
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className='flex flex-col mt-7 text-center justify-center'>
      <Image />
      <p><strong>Username:</strong> {profile?.username}</p>
      <p><strong>Email:</strong> {profile?.email}</p>
    </div>
  );
}
