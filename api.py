from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from predict import predict

app = FastAPI(title="Helpdesk ML", version="1.0.0")

class Ticket(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Texto vazio")
        return value

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def classify(ticket: Ticket):
    return predict(ticket.text)
