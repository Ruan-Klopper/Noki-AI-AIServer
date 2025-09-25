from fastapi import FastAPI
from app.routes import chat, schedule

app = FastAPI(title="Noki AI Engine")

# Include routers
app.include_router(chat.router)
app.include_router(schedule.router)

@app.get("/")
def health_check():
    return {"status": "ok"}
