# Publicação do Helpdesk ML no Hugging Face

O código está no GitHub. A publicação do modelo e da demo é a próxima etapa.

## Modelo

1. Entre na conta JuliaNeris0017 e crie um novo modelo chamado helpdesk-tfidf-pt.
2. Em Files and versions, escolha Add file e Upload files.
3. Envie os três arquivos da pasta model na raiz do repositório de modelo.
4. Confirme o commit e confira o model card.

## Modelo e demo por script

Na pasta do projeto, após instalar requirements.txt, execute:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import login; login()"
.venv\Scripts\python.exe publish.py --repo JuliaNeris0017/helpdesk-tfidf-pt --space JuliaNeris0017/helpdesk-ml-demo
```

Forneça um token de escrita no campo protegido do login. O script cria os destinos e envia os arquivos, ou atualiza os destinos existentes. Aguarde o build do Space antes de divulgar a demonstração. O modelo clássico roda em CPU.

## Verificação após publicar

- Confira se model.joblib, metrics.json e README.md estão no repositório do modelo.
- Abra a demo e envie uma mensagem de exemplo.
- Confira se o retorno informa a categoria sugerida e a necessidade de revisão.
- Acrescente os endereços confirmados ao README e ao site do portfólio.

O modelo clássico e o adaptador LoRA têm repositórios separados. Para gerar o adaptador, execute primeiro Fine_tuning_Colab.ipynb em GPU. O notebook inclui as instruções de publicação após o treinamento.

Referências

- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://huggingface.co/docs/hub/spaces-sdks-gradio
