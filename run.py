import uvicorn
import sys
import asyncio

if __name__ == "__main__":
    # 1. FORCE THE FIX: Apply Proactor Loop BEFORE Uvicorn ever wakes up
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("🔧 Windows Proactor Loop Policy Applied (Unlimited Connections)")

    # 2. START SERVER PROGRAMMATICALLY
    # We disable the separate 'reload' process to keep our Loop Policy active
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False  # <--- IMPORTANT: Set to False for stability on Windows
    )