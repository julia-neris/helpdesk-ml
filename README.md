# Helpdesk ML · Classificação de chamados de TI

Protótipo de Machine Learning que sugere uma categoria para chamados em português e disponibiliza a previsão por uma API FastAPI e uma interface Gradio. O projeto inclui dados sintéticos, treinamento reproduzível e avaliação por cenário.

**Autora do portfólio — Júlia Barbosa Neris.** Código, dados sintéticos e documentação preparados com assistência de IA.

[Artigo técnico](docs/artigo.md) · [Resultados completos](model/metrics.json) · [Dados](data/tickets.json) · [Publicação no Hugging Face](docs/publicacao.md)

## Problema e solução

Uma equipe de suporte precisa direcionar mensagens para filas de atendimento. Este experimento transforma o texto em atributos TF-IDF e usa regressão logística para recomendar uma das categorias abaixo. Toda sugestão requer revisão humana.

| Categoria | Exemplo de assunto |
|---|---|
| Acesso | Senhas, autenticação e permissões |
| Rede | Wi-Fi, VPN e conectividade |
| Software | Instalação, falhas e licenças de aplicativos |
| Hardware | Teclado, bateria, monitor e periféricos |
| Dados | Atualização de dashboards, consultas e qualidade dos dados |

## Resultado e metodologia

O conjunto contém **80 mensagens sintéticas em 40 cenários**, com duas formulações por cenário. As paráfrases permanecem juntas na separação entre treino, validação e teste.

| Conjunto | Mensagens | Uso |
|---|---:|---|
| Treino | 50 | Ajustar o vetorizador e o classificador |
| Validação | 10 | Selecionar o modelo pelo macro-F1 |
| Teste | 20 | Avaliar o modelo selecionado |

| Modelo | Macro-F1 na validação |
|---|---:|
| Classe majoritária | 0,0667 |
| TF-IDF de palavras + regressão logística | 0,1800 |
| TF-IDF de caracteres + regressão logística | **0,2600** |

O modelo de caracteres obteve **macro-F1 de 0,5400 e acurácia de 55% no teste**. São 11 acertos em 20 mensagens. O artefato publicado mantém o treinamento original, sem reajuste com validação ou teste.

Esses resultados descrevem um piloto pequeno e sintético. A aplicação sempre sugere uma das cinco categorias, inclusive quando a mensagem está fora do domínio. Mensagens com múltiplas intenções, categorias novas e chamados reais precisam de avaliação específica. O campo de revisão humana é obrigatório em todas as respostas.

## Executar no Windows

Requer Python 3.11 ou 3.12. Baixe o repositório pelo botão Code → Download ZIP e extraia, ou clone com Git. Abra o terminal na pasta extraída.

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Abra o endereço local exibido no terminal. O modelo treinado já está em model/model.joblib.

Para reproduzir o treinamento e as métricas:

```powershell
.venv\Scripts\python.exe train.py
```

Esse comando substitui model/model.joblib e model/metrics.json com a nova execução.

Em Linux ou macOS, crie o ambiente com python3 -m venv .venv e use .venv/bin/python nos mesmos comandos.

## Executar a API

```powershell
.venv\Scripts\python.exe -m uvicorn api:app --reload
```

Abra http://127.0.0.1:8000/docs. Em POST /predict, selecione Try it out e envie:

```json
{"text": "Esqueci minha senha"}
```

Resposta de exemplo:

```json
{"categoria_sugerida": "acesso", "revisao_humana": true}
```

A API rejeita mensagens vazias, compostas somente de espaços ou maiores que 4.000 caracteres. GET /health informa que o serviço está em execução.

## Arquivos principais

| Arquivo | Finalidade |
|---|---|
| train.py | Comparação de modelos e avaliação |
| predict.py | Carregamento do pipeline e validação das entradas |
| api.py | API HTTP |
| app.py | Interface Gradio |
| data/tickets.json | Mensagens, categorias, cenários e splits |
| model/ | Modelo treinado, métricas e documentação |
| train_lora.py | Experimento de fine-tuning do Qwen com LoRA |
| Fine_tuning_Colab.ipynb | Execução do experimento LLM no Colab |
| publish.py | Publicação de modelo e Space no Hugging Face |

## Extensão com LLM

O código de LoRA está preparado para comparar Qwen2.5-0.5B-Instruct antes e depois do ajuste, com três épocas fixadas previamente. A avaliação mede macro-F1, saídas válidas e latência. Esse experimento ainda requer treinamento em GPU. O repositório contém apenas os pesos do modelo clássico.

Para usar o notebook, compacte o conteúdo deste repositório em um ZIP, abra Fine_tuning_Colab.ipynb no Colab, ative GPU e siga as células. As células finais publicam o adaptador na sua conta após autenticação. Revise os resultados antes de executá-las.

## Verificação e próximos passos

O treinamento, as métricas e a API foram verificados novamente antes da publicação. Consulte [o registro de validação](docs/validacao.md). A demo Gradio e o treinamento LoRA ainda precisam de execução no ambiente de destino.

As próximas etapas são publicar o modelo e a demo no Hugging Face, ampliar a variedade de cenários e avaliar chamados reais autorizados com rótulos revisados.
