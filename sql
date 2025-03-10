create database SuperTrend;
CREATE TABLE sessions(
sesID ,
userType ,
activity ,
id INT AUTO_INCREMENT PRIMARY KEY,   --Foreign Key




);

CREATE TABLE tokenmap(
sesID -- foreign key
token --timeout
uid --foreign key(users)
);

CREATE TABLE UserQuery(
uid --foreign key
req
res
);

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,          -- Auto-incremented unique user ID
    first_name VARCHAR(255) NOT NULL,           -- User's first name
    last_name VARCHAR(255) NOT NULL,            -- User's last name
    email VARCHAR(255) NOT NULL UNIQUE,         -- User's email (must be unique)
    password VARCHAR(255) NOT NULL,             -- Hashed password
    marketing_accept BOOLEAN DEFAULT FALSE,     -- Marketing consent flag (default to false)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Timestamp of when the account was created
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP -- Timestamp for when account was last updated
);

-- Optionally, add an index on the email column to improve lookups:
CREATE INDEX idx_email ON users (email);

