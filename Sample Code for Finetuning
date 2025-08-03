# Sample Code Finetuning v1 — Workflow

1. **Install libraries** – one Colab-friendly cell covers `unsloth`, `trl`, etc.  
2. **Load base model** – pull TigerLLM (or any Gemma-3 ckpt) in 8-bit.  
3. **Add LoRA adapters** – `FastModel.get_peft_model` sets trainable modules.  
4. **Prepare data**  
   * Read `trial.csv` → map to `conversations` schema.  
   * Apply Gemma-3 chat template, build plain-text `text` field.  
   * Verify formatting & masking.  
5. **Configure SFT** – set batch size, epochs, LR, gradient accumulation.  
6. **Finetune** – launch `SFTTrainer`, monitor GPU stats, capture `trainer_stats`.  
7. **Evaluate** – prompt the finetuned model + run batch inference on dev set, persisting `response` column and valid-fence %.  
8. **Save** – write the trained weights & tokenizer to `New_Model/`.

_Output: a finetuned TigerLLM directory and a CSV with fresh code answers._
