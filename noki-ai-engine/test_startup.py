#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI application can start
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import fastapi
        print("✓ FastAPI imported successfully")
        
        import uvicorn
        print("✓ Uvicorn imported successfully")
        
        # Test app import
        from app.main import app
        print("✓ FastAPI app imported successfully")
        
        # Test health endpoint
        from app.routes.health import router
        print("✓ Health router imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_start():
    """Test if the app can be created"""
    try:
        print("\nTesting app creation...")
        from app.main import app
        
        # Test basic app properties
        print(f"✓ App title: {app.title}")
        print(f"✓ App version: {app.version}")
        print(f"✓ App debug: {app.debug}")
        
        print("\n✅ App creation successful!")
        return True
        
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Noki AI Engine startup...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_app_start()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The application should start successfully.")
        sys.exit(0)
    else:
        print("💥 Tests failed! Check the errors above.")
        sys.exit(1)
