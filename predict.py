from pathlib import Path
from functools import lru_cache
import joblib

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(Path(__file__).resolve().parent / "model/model.joblib")

def predict(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Digite uma mensagem não vazia.")
    if len(text) > 4000:
        raise ValueError("Use até 4000 caracteres.")
    # Protótipo de recomendação. Todos os encaminhamentos exigem revisão.
    return {"categoria_sugerida": str(load_model().predict([text])[0]),
            "revisao_humana": True}
