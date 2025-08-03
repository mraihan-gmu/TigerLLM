# Sample Code Prompting v1 — Workflow

1. **Setup**  
   * Import Unsloth + supporting libs.  
   * Attach Gemma-3 chat template to tokenizer.

2. **Model Loading**  
   * Load TigerLLM (or any Gemma-3-style checkpoint) in 8-bit via `FastModel.from_pretrained`.

3. **Single-Prompt Demo**  
   * Feed one Bangla question.  
   * Generate a fenced ```python``` solution.

4. **Batch Generation**  
   * Read `dev.csv` (`id, instruction`).  
   * Generate `response` for each row.  
   * Validate strict code-block fencing; leave cell blank if missing.  
   * Write results to `dev_with_response.csv`.

5. **Reporting**  
   * Print the percentage of rows containing valid fenced code.

_Output → a clean CSV ready for evaluation._
