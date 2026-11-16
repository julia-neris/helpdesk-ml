# Classificação de chamados de TI com TF-IDF e avaliação por cenário

Texto técnico preparado para revisão e publicação por Júlia Barbosa Neris, com assistência de IA.
Experimento executado em 7 de novembro de 2026.

A triagem de chamados pode organizar a fila de uma equipe de suporte. Neste experimento, a tarefa foi sugerir uma de cinco categorias — acesso, rede, software, hardware e dados — para uma mensagem em português. A implementação está na pasta helpdesk-ml.

A pergunta experimental foi se representações de palavras ou de fragmentos de caracteres generalizariam melhor para cenários novos dentro dessas categorias. Foram criadas 80 mensagens sintéticas, organizadas em 40 cenários. Cada cenário tem duas formulações. A separação por cenário mantém paráfrases próximas no mesmo conjunto e reduz uma fonte de vazamento entre treinamento e avaliação.

O treino contém 50 mensagens, a validação 10 e o teste 20. A validação seleciona entre classe majoritária, TF-IDF de palavras com regressão logística e TF-IDF de caracteres com regressão logística. O modelo escolhido mantém o ajuste feito apenas no treino.

Os macro-F1 de validação foram {"majority": 0.06666666666666667, "word_tfidf": 0.18, "char_tfidf": 0.26}. O modelo selecionado foi char_tfidf. No teste, o macro-F1 foi 0.5400 e a acurácia foi 0.5500. Macro-F1 dá o mesmo peso a cada categoria. O relatório inclui precisão, revocação, F1 por classe e todas as previsões, permitindo examinar erros individualmente.

O conjunto é pequeno e inteiramente sintético. A distribuição de termos pode refletir escolhas de quem redigiu os exemplos. A avaliação descreve apenas este piloto. Uma avaliação operacional exigiria chamados reais autorizados, revisão dos rótulos por especialistas e um teste independente com textos ambíguos e assuntos fora de escopo. A API atual exige revisão humana para toda sugestão.

O próximo experimento está implementado em train_lora.py. Ele compara o Qwen2.5-0.5B-Instruct via prompt com um adaptador LoRA, usando os mesmos splits. O script mede macro-F1, taxa de respostas válidas e latência. Esse treinamento ainda precisa ser executado em GPU, por isso este artigo registra resultados apenas dos modelos clássicos.

Para reproduzir, execute train.py e consulte model/metrics.json. O pipeline inclui vetorização e classificação no mesmo artefato, evitando diferenças entre o pré-processamento do treino e da inferência. A API valida o comprimento da entrada e rejeita mensagens vazias.

Referências

- https://scikit-learn.org/stable/modules/feature_extraction.html
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- https://huggingface.co/docs/peft/quicktour
