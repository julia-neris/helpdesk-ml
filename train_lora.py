"""Experimento LoRA. Requer download do Qwen e GPU CUDA. Sem resultados pré-fabricados."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
LABELS = ["acesso", "dados", "hardware", "rede", "software"]

def prompt(tokenizer, text):
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Classifique o chamado de TI. Responda somente com uma categoria: " + ", ".join(LABELS) + "."},
        {"role": "user", "content": text},
    ], tokenize=False, add_generation_prompt=True)

def evaluate(model, tokenizer, rows):
    model.eval()
    records = []
    for row in rows:
        inputs = tokenizer(prompt(tokenizer, row["text"]), return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        predicted = answer if answer in LABELS else "invalid"
        records.append(dict(text=row["text"], expected=row["label"], predicted=predicted,
                            raw_answer=answer, seconds=elapsed))
    return {
        "macro_f1": f1_score([r["expected"] for r in records], [r["predicted"] for r in records],
                             labels=LABELS, average="macro", zero_division=0),
        "valid_output_rate": sum(r["predicted"] != "invalid" for r in records) / len(records),
        "latency_median_seconds": float(np.median([r["seconds"] for r in records])),
        "records": records,
    }

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU CUDA no Colab ou Kaggle antes de executar.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    rows = json.loads((ROOT / "data/tickets.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to("cuda")
    base.config.use_cache = False
    before_validation = evaluate(base, tokenizer, validation)
    before_test = evaluate(base, tokenizer, test)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj"],
                                           task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for epoch in range(3):
        model.train()
        random.shuffle(train)
        epoch_losses = []
        for row in train:
            prefix = tokenizer(prompt(tokenizer, row["text"]), add_special_tokens=False)["input_ids"]
            target = tokenizer(row["label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = prefix + target
            if len(ids) > 512:
                raise ValueError("Exemplo maior que 512 tokens. Revise o texto antes de treinar.")
            input_ids = torch.tensor([ids], device="cuda")
            labels = torch.tensor([[-100] * len(prefix) + target], device="cuda")
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
        print(f"Época {epoch + 1}: loss={losses[-1]:.4f}")
    adapter = ROOT / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    results = dict(base_model=BASE, seed=42, epochs=3, train_losses=losses,
                   selection="3 épocas fixadas antes do experimento; sem seleção pelo teste",
                   before_validation=before_validation, before_test=before_test,
                   after_validation=evaluate(model, tokenizer, validation),
                   after_test=evaluate(model, tokenizer, test),
                   hardware=torch.cuda.get_device_name())
    (adapter / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    card = f"""---
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
language: pt
license: apache-2.0
---
# Helpdesk LoRA

Adaptador experimental treinado em 50 mensagens sintéticas de TI. Requer o modelo base {BASE}.
Validação com 10 mensagens e teste com 20 mensagens, separados por cenário.
Três épocas, seed 42, r=8, alpha=16, dropout=0.05, módulos q_proj e v_proj.

Macro-F1 antes: {results['before_test']['macro_f1']:.4f}.
Macro-F1 depois: {results['after_test']['macro_f1']:.4f}.
Consulte metrics.json para previsões, saídas inválidas e latência.
Os resultados deste conjunto sintético pequeno não estimam desempenho em chamados reais.
Textos com múltiplas intenções e fora do domínio exigem revisão humana.
Treinamento e documentação preparados com assistência de IA.

## Carregar
Use AutoTokenizer.from_pretrained com este repositório e
AutoModelForCausalLM.from_pretrained com o modelo base.
Aplique PeftModel.from_pretrained(base, identificador_deste_repositorio).
Use o template de chat e o prompt registrados em train_lora.py.
"""
    (adapter / "README.md").write_text(card, encoding="utf-8")

if __name__ == "__main__":
    main()
