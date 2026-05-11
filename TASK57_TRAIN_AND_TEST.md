# 🚀 Treinar Checkpoints + Rodar Generalization Test

**Problema:** O generalization test falhou porque faltam checkpoints dos modelos treinados.

**Solução:** Executar o workflow que treina os checkpoints e depois roda o test.

---

## ⚡ Opção 1: Workflow Completo (Recomendado)

Treina checkpoints + roda generalization test em um comando:

```bash
cd /home/wesleyferreiramaia/data/sidewalk-accessibility-project

sbatch src/generalization/workflow_train_and_test.sh

# Monitorar progresso
squeue -u $USER
tail -f logs/workflow_*.log

# Verificar resultados quando terminar
cat results/generalization/dinov2-large/predictions.csv | head
cat results/generalization/dinov2-large/agreement_summary.json
```

**Tempo esperado:** ~15-20 minutos (treina 5 aids + avalia 30 imagens)

**Output:**
```
results/models/dinov2-large/
├── walking_cane/
│   ├── probe.pth          ← Modelo treinado
│   └── scaler.joblib      ← Normalizador
├── walker/
│   └── ...
└── ... (5 aids total)

results/generalization/dinov2-large/
├── predictions.csv        ← Predições
└── agreement_summary.json ← Estatísticas
```

---

## ⚡ Opção 2: Treinar Apenas (sem test)

Se quiser treinar os checkpoints separadamente:

```bash
python src/models/train_final_model.py \
    --encoder dinov2-large \
    --output_dir results/models/dinov2-large \
    --loss_type soft_kl
```

**Output:** Checkpoints em `results/models/dinov2-large/`

Depois rodar:
```bash
python src/generalization/evaluate_generalization.py \
    --encoder dinov2-large \
    --checkpoint results/models/dinov2-large \
    --test_images data/generalization/test_images.csv \
    --output_dir results/generalization/dinov2-large \
    --use_wandb
```

---

## 📊 Metodologia (Match Paper Section 3)

O script segue **exatamente** a metodologia do paper:

1. **Carrega distribuições** de votos de 829 usuários (`tallies_firebase.json`)
2. **Extrai features** com encoder congelado (L2-normalized)
3. **Treina probe linear** por mobility aid (5 total)
4. **Usa Soft-KL loss** contra as distribuições de votos (não hard labels)
5. **Salva checkpoints** em `results/models/[encoder]/[aid]/`

---

## 🎯 Checklist

- [ ] Dados carregados: `data/processed/tallies_firebase.json`
- [ ] Imagens existem: `data/images/sidewalk-images/`
- [ ] Test images prontas: `data/generalization/test_images.csv`
- [ ] GPU disponível: `nvidia-smi`
- [ ] W&B login (uma vez só): `wandb login`
- [ ] Rodar workflow: `sbatch src/generalization/workflow_train_and_test.sh`
- [ ] Ver logs: `tail -f logs/workflow_*.log`
- [ ] Verificar resultados: `cat results/generalization/dinov2-large/predictions.csv`

---

## 📈 W&B Tracking (Automático)

O workflow rastreia automaticamente em W&B:
- Accuracy per city
- Balanced accuracy
- Confusion matrix
- Hardware (GPU, CUDA)
- Tempo total

Ver em: https://wandb.ai/seu-usuario/sidewalk-generalization

---

## 🆘 Troubleshooting

**"tallies_firebase.json not found"**
```bash
# Verificar se dados existem
ls -la data/processed/tallies_firebase.json
```

**"images not found"**
```bash
# Verificar se imagens existem
ls data/images/sidewalk-images/ | head
```

**"Out of memory"**
- Reduzir batch size em train_final_model.py
- Ou aumentar `--mem=` no SLURM

**"Train rodou mas predictions vazios"**
- Checar se os checkpoints foram salvos: `ls results/models/dinov2-large/*/probe.pth`
- Checar se CSV de test images tem caminho correto
- Ver logs completos: `cat logs/workflow_*.log`

---

## 📚 Arquivos

| Arquivo | Função |
|---------|--------|
| `src/models/train_final_model.py` | Treina checkpoints (paper methodology) |
| `src/generalization/evaluate_generalization.py` | Avalia em test images |
| `src/generalization/workflow_train_and_test.sh` | SLURM workflow completo |

---

**Pronto! Execute e aguarde ~20 minutos. 🚀**
