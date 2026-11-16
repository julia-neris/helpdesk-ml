# Verificação da publicação

Executada em 16 de novembro de 2026.

- Treinamento reproduzido em cópia temporária. Todas as métricas e previsões coincidiram com o relatório original.
- As 20 previsões do artefato salvo coincidiram com model/metrics.json.
- GET /health e POST /predict verificados com FastAPI TestClient.
- Texto vazio, somente espaços e acima de 4.000 caracteres rejeitado com HTTP 422.
- Sintaxe dos scripts e das células Python do notebook verificada.

Ambiente de avaliação — Python 3.12, scikit-learn 1.8.0, NumPy 2.3.5 e joblib 1.5.3.

A interface Gradio, o upload para o Hugging Face e o treinamento LoRA não foram executados nesta verificação.
