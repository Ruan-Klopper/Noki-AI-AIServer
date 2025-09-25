# Environment Setup Guide

## Overview

This project uses environment variables for configuration. Follow these steps to set up your environment properly.

## Step 1: Create Environment File

Create a `.env` file in the root directory (`/Users/ruanklopper/Documents/Noki AI/Project/Noki-AI-AIServer/`) with the following content:

```bash
# FastAPI Configuration
APP_NAME="Noki AI Engine"
APP_VERSION="1.0.0"
DEBUG=True
HOST="0.0.0.0"
PORT=8000

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Database Configuration
DATABASE_URL=sqlite:///./noki_ai.db

# Supabase Configuration (if using Supabase)
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# LangChain Configuration
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=noki-ai-engine

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:8080"]

# Logging
LOG_LEVEL=INFO
```

## Step 2: Update Configuration Values

Replace the placeholder values with your actual configuration:

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL (if using Supabase)
- `SUPABASE_KEY`: Your Supabase anon key (if using Supabase)
- `LANGCHAIN_API_KEY`: Your LangChain API key (if using LangChain)
- `SECRET_KEY`: A secure secret key for JWT tokens

## Step 3: Verify .gitignore

The `.gitignore` file has been updated to include:

- `.env` files (to prevent committing sensitive data)
- Python-specific ignores
- Virtual environment ignores
- IDE and OS-specific ignores

## Step 4: Install Dependencies

Make sure you have the required dependencies installed:

```bash
cd noki-ai-engine
pip install python-dotenv pydantic-settings
```

## Configuration Management

The application uses the `config.py` file to manage environment variables with Pydantic settings. This provides:

- Type validation
- Default values
- Environment variable loading
- Configuration documentation

## Security Notes

- Never commit `.env` files to version control
- Use different `.env` files for different environments (development, staging, production)
- Keep your API keys secure and rotate them regularly
- Use strong, unique secret keys for production
