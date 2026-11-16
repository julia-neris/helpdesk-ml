---
library_name: sklearn
language: pt
tags:
- scikit-learn
- text-classification
- tf-idf
- logistic-regression
---
# Helpdesk ML · Modelo de classificação

Pipeline TF-IDF de caracteres com regressão logística, treinado com 50 mensagens sintéticas em português. As categorias são acesso, dados, hardware, rede e software.

## Dados e avaliação

O conjunto completo contém 80 mensagens e 40 cenários. Cada cenário tem duas formulações, mantidas no mesmo split. A validação tem 10 mensagens e seleciona o modelo pelo macro-F1. O teste tem 20 mensagens.

- Macro-F1 na validação do modelo escolhido — 0,2600.
- Macro-F1 no teste — 0,5400.
- Acurácia no teste — 0,5500, equivalente a 11 de 20 mensagens.

O modelo mantém o ajuste feito somente no treino. As métricas e todas as previsões estão em metrics.json.

## Carregar o artefato

Instale scikit-learn==1.8.0, numpy==2.3.5 e joblib==1.5.3. Baixe model.joblib para o diretório atual.

```python
import joblib

model = joblib.load("model.joblib")
print(model.predict(["Esqueci minha senha"]).tolist())
```

Carregue arquivos joblib somente de fontes confiáveis, pois a desserialização pode executar código.

## Limitações e uso pretendido

Este é um protótipo de portfólio com dados sintéticos. Não foi validado em produção. O classificador sempre retorna uma das cinco categorias, mesmo para mensagens fora do domínio. Todas as sugestões exigem revisão humana. A avaliação pequena é insuficiente para estimar desempenho operacional.

Os dados, o código e a documentação foram preparados com assistência de IA. A execução inicial ocorreu em 7 de novembro de 2026 e a reprodução foi verificada em 16 de novembro de 2026.

## Código e dados

https://github.com/julia-neris/helpdesk-ml
