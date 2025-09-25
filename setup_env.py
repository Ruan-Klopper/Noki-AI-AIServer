#!/usr/bin/env python3
"""
Environment setup script for Noki AI Engine
This script helps create a .env file with the necessary configuration.
"""

import os
from pathlib import Path

def create_env_file():
    """Create a .env file with default configuration"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    env_file_path = project_root / ".env"
    
    # Check if .env already exists
    if env_file_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Setup cancelled.")
            return
    
    # Default environment configuration
    env_content = """# Noki AI Engine Environment Configuration
# Update these values with your actual configuration

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
"""
    
    try:
        # Write the .env file
        with open(env_file_path, 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created successfully!")
        print(f"📁 Location: {env_file_path}")
        print("\n📝 Next steps:")
        print("1. Edit the .env file and replace placeholder values with your actual configuration")
        print("2. Make sure to add your API keys (OpenAI, Supabase, etc.)")
        print("3. Change the SECRET_KEY to a secure random string")
        print("4. The .env file is already included in .gitignore for security")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

if __name__ == "__main__":
    print("🚀 Noki AI Engine Environment Setup")
    print("=" * 40)
    create_env_file()
