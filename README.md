<div align="center">
  <h1 style="text-align: center; color: green;"> Accepted in ACL Main 2025 </h1>
</div>

<div align="center">
<table>
<tr>
<td>
<a href="https://arxiv.org/pdf/2503.10995">
<img src="https://img.shields.io/badge/arXiv-Read_Paper-blue?style=for-the-badge&logo=arxiv" alt="Read Paper"/>
</a>
</td>
<td>
<a href="[https://huggingface.co/datasets/md-nishat-008/mHumanEval-Benchmark](https://huggingface.co/md-nishat-008/TigerLLM-1B-it](https://huggingface.co/md-nishat-008/TigerLLM-1B-it)">
<img src="https://img.shields.io/badge/HuggingFace-mHumanEval_Benchmark-orange?style=for-the-badge&logo=huggingface" alt="mHumanEval"/>
</a>
</td>
<td>
<a href="mailto:mraihan2@gmu.edu">
<img src="https://img.shields.io/badge/Email-Contact_Us-blue?style=for-the-badge&logo=gmail" alt="Contact Us"/>
</a>
</td>
</tr>
</table>
</div>

<div align="center">

<h1 style="text-align: center; color: green;">TigerLLM - A Family of Bangla Large Language Models</h1>

<h3 style="text-align: center; color: green;">Nishat Raihan, Marcos Zampieri</h3>
<h4 style="text-align: center; color: green;">George Mason University, VA, USA</h4>
<p style="text-align: center; color: red;">mraihan2@gmu.edu</p>

</div>


---
If you find our work helpful, please consider citing our paper:

```bibtex
@article{raihan2025tigerllm,
  title={TigerLLM-A Family of Bangla Large Language Models},
  author={Raihan, Nishat and Zampieri, Marcos},
  journal={arXiv preprint arXiv:2503.10995},
  year={2025}
}
```





<hr>

<h2 style="text-align: center; color: green;">Abstract</h2>
<p>
The development of Large Language Models (LLMs) remains heavily skewed towards English and a few other high-resource languages. This linguistic disparity is particularly evident for Bangla – the 5th most spoken language. A few initiatives attempted to create open-source Bangla LLMs with performance still behind high-resource languages and limited reproducibility. To address this gap, we introduce <span style="color: red;">TigerLLM</span> – a family of Bangla LLMs. Our results demonstrate that these models surpass all open-source alternatives and also outperform larger proprietary models like GPT3.5 across standard benchmarks, establishing TigerLLM as the new baseline for future Bangla language modeling.
</p>

<hr>

<h2 style="text-align: center; color: green;">1. Introduction</h2>
<p>
LLMs have fundamentally transformed NLP by achieving exceptional performance across a wide range of tasks. However, their advancements have predominantly benefited high-resource languages. Despite having about 237 million native Bangla speakers, Bangla remains underserved in modern NLP due to the lack of high-quality training data and reproducible methodologies.
</p>

<h3 style="text-align: center; color: green;">1.1 Limitations of Bangla LLM Initiatives</h3>
<p>
Recent efforts (e.g., titu-Gemma, titu-LLaMA, Bangla-LLaMA, G2B) suffer from low reproducibility, suboptimal performance, and poor documentation. Many rely on translated synthetic datasets, leading to compromised instruction quality.
</p>

<table>
  <thead>
    <tr>
      <th style="color: green; text-align: center;">Base-LLM</th>
      <th style="color: green; text-align: center;">Size</th>
      <th style="color: green; text-align: center;">Pretraining<br>(pt)</th>
      <th style="color: green; text-align: center;">Corpora</th>
      <th style="color: green; text-align: center;">Finetuning<br>(ft)</th>
      <th style="color: green; text-align: center;">Finetune Dataset</th>
      <th style="color: green; text-align: center;">Paper/Report?</th>
      <th style="color: green; text-align: center;">Reproducibility?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>titu-Gemma (Gemma-2)</td>
      <td>2B</td>
      <td>4.4B</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
    </tr>
    <tr>
      <td>titu-LLaMA (LLaMA-3.1)</td>
      <td>3B</td>
      <td>37B</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
    </tr>
    <tr>
      <td>Bangla-LLaMA (LLaMA-3.2)</td>
      <td>3B</td>
      <td>&#10003;</td>
      <td>&#10005;</td>
      <td>172K<br>(Orca-translated)</td>
      <td>&#10003;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
    </tr>
    <tr>
      <td>G2B (Gemma-2)</td>
      <td>9B</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>145K<br>(Alpaca-translated)</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
    </tr>
    <tr>
      <td>Bangla-LLaMA (LLaMA-2)</td>
      <td>13B</td>
      <td>&#10003;</td>
      <td>&#10005;</td>
      <td>145K<br>(Alpaca-translated)</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
      <td>&#10005;</td>
    </tr>
    <tr>
      <td><span style="color:red;">TigerLLM (LLaMA-3.2)</span></td>
      <td>1B</td>
      <td>10M</td>
      <td>Bangla-TextBook</td>
      <td>100K<br>(Bangla-Instruct)</td>
      <td>&#10003;</td>
      <td>&#10003;</td>
    </tr>
    <tr>
      <td><span style="color:red;">TigerLLM (Gemma-2)</span></td>
      <td>9B</td>
      <td>10M</td>
      <td>Bangla-TextBook</td>
      <td>100K<br>(Bangla-Instruct)</td>
      <td>&#10003;</td>
      <td>&#10003;</td>
    </tr>
  </tbody>
</table>

<h3 style="text-align: center; color: green;">1.2 Contributions</h3>
<ul>
  <li><span style="color: red;">Bangla-TextBook Corpus</span>: A 10M-token corpus of high-quality educational texts.</li>
  <li><span style="color: red;">Bangla-Instruct Dataset</span>: 100K native Bangla instruction-response pairs generated via self-instruct and advanced teacher models.</li>
  <li><span style="color: red;">TigerLLM Models</span>: A family of models (1B and 9B parameters) that achieve significant performance improvements over existing alternatives.</li>
</ul>

<hr>

<h2 style="text-align: center; color: green;">2. Bangla-TextBook Corpus</h2>
<p>
The <span style="color: red;">Bangla-TextBook</span> corpus is compiled exclusively from open-source educational materials provided by the National Curriculum and Textbook Board of Bangladesh. It aggregates texts from <span style="color: red;">163 textbooks</span> for Grades 6–12, yielding <span style="color: red;">9,897,623 tokens</span> and <span style="color: red;">697,903 sentences</span>, capturing authentic academic language use.
</p>

<hr>

<h2 style="text-align: center; color: green;">3. Bangla-Instruct</h2>
<p>
To overcome previous limitations, the <span style="color: red;">Bangla-Instruct</span> dataset contains <span style="color: red;">100,000 instruction-response pairs</span> generated using a self-instruct framework. Key steps include:
</p>
<ol>
  <li><span style="color: red;">Seed Task Generation</span>: 500 tasks curated by 50 volunteers from diverse academic backgrounds.</li>
  <li>New instruction generation using GPT-4 and Claude-3.5-Sonnet.</li>
  <li>Task identification for appropriate response formatting.</li>
  <li>Multi-stage filtering to ensure linguistic quality and cultural sensitivity.</li>
</ol>
<p>
Refer to <span style="color: red;">Figure 1</span> for the Bangla-Instruct generation pipeline.
</p>

<hr>

<h2 style="text-align: center; color: green;">4. TigerLLM</h2>
<p>
TigerLLM is built by leveraging the strengths of both the Bangla-TextBook corpus and the Bangla-Instruct dataset. The training process involves:
</p>
<ul>
  <li><span style="color: red;">Continual Pretraining</span> on the Bangla-TextBook corpus to capture language-specific nuances.</li>
  <li><span style="color: red;">Model Distillation</span> via full fine-tuning (without LoRA) using Flash Attention, ensuring efficient convergence.</li>
</ul>
<p>
For details on the training pipeline, please see <span style="color: red;">Figure 2</span> (overall pipeline), <span style="color: red;">Figure 3</span> (pretraining loss), and <span style="color: red;">Figure 4</span> (finetuning loss).
</p>

<hr>

<h2 style="text-align: center; color: green;">5. Evaluation</h2>
<p>
TigerLLM is evaluated on multiple Bangla-specific benchmarks including:
</p>
<ul>
  <li>MMLU-bn</li>
  <li>PangBench-bn</li>
  <li>BanglaQuaD</li>
  <li>mHumanEval-bn</li>
  <li>BEnQA</li>
  <li>BanglaRQA</li>
</ul>
<p>
The performance comparison is detailed in <span style="color: red;">Table 2</span> below:
</p>

<table>
  <thead>
    <tr>
      <th style="color: green; text-align: center;">Model</th>
      <th style="color: green; text-align: center;">MMLU-bn</th>
      <th style="color: green; text-align: center;">PangBench-bn</th>
      <th style="color: green; text-align: center;">BanglaQuaD</th>
      <th style="color: green; text-align: center;">mHumanEval-bn</th>
      <th style="color: green; text-align: center;">BEnQA</th>
      <th style="color: green; text-align: center;">BanglaRQA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GPT3.5</td>
      <td>0.55</td>
      <td>0.55</td>
      <td>0.50</td>
      <td>0.56</td>
      <td>0.50</td>
      <td>0.49</td>
    </tr>
    <tr>
      <td>Gemini-Flash1.5</td>
      <td>0.66</td>
      <td>0.57</td>
      <td>0.62</td>
      <td>0.58</td>
      <td>0.56</td>
      <td>0.61</td>
    </tr>
    <tr>
      <td>GPT4o-mini</td>
      <td>0.67</td>
      <td>0.62</td>
      <td>0.65</td>
      <td>0.56</td>
      <td>0.60</td>
      <td>0.60</td>
    </tr>
    <tr>
      <td>LLaMA3.2 (11B)</td>
      <td>0.22</td>
      <td>0.19</td>
      <td>0.21</td>
      <td>0.15</td>
      <td>0.18</td>
      <td>0.20</td>
    </tr>
    <tr>
      <td>Gemma 2 (27B)</td>
      <td>0.35</td>
      <td>0.51</td>
      <td>0.43</td>
      <td>0.64</td>
      <td>0.50</td>
      <td>0.56</td>
    </tr>
    <tr>
      <td>Pangea (7B)</td>
      <td>0.18</td>
      <td>0.15</td>
      <td>0.17</td>
      <td>0.10</td>
      <td>0.14</td>
      <td>0.16</td>
    </tr>
    <tr>
      <td><span style="color:red;">Titu-LLM</span></td>
      <td>0.06</td>
      <td>0.19</td>
      <td>0.08</td>
      <td>0.02</td>
      <td>0.17</td>
      <td>0.21</td>
    </tr>
    <tr>
      <td><span style="color:red;">Bong-LLaMA</span></td>
      <td>0.05</td>
      <td>0.12</td>
      <td>0.08</td>
      <td>0.02</td>
      <td>0.15</td>
      <td>0.13</td>
    </tr>
    <tr>
      <td><span style="color:red;">Bangla-LLaMA</span></td>
      <td>0.02</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.10</td>
      <td>0.11</td>
      <td>0.09</td>
    </tr>
    <tr>
      <td><span style="color:red;">Bangla-Gemma</span></td>
      <td>0.18</td>
      <td>0.15</td>
      <td>0.12</td>
      <td>0.10</td>
      <td>0.22</td>
      <td>0.19</td>
    </tr>
    <tr>
      <td><span style="color:red;">TigerLLM (1B)</span></td>
      <td>0.61</td>
      <td>0.55</td>
      <td>0.68</td>
      <td>0.61</td>
      <td>0.59</td>
      <td>0.62</td>
    </tr>
    <tr>
      <td><span style="color:red;">TigerLLM (9B)</span></td>
      <td>0.72</td>
      <td>0.68</td>
      <td>0.70</td>
      <td>0.63</td>
      <td>0.65</td>
      <td>0.68</td>
    </tr>
  </tbody>
</table>

<hr>

<h2 style="text-align: center; color: green;">6. Conclusion and Future Work</h2>
<p>
This paper presents <span style="color: red;">TigerLLM</span>, a family of Bangla language models that set new benchmarks by leveraging two high-quality datasets: the Bangla-TextBook corpus and the Bangla-Instruct dataset. Future work will involve qualitative analyses, expanding the corpus, scaling model sizes, and developing more sophisticated evaluation metrics.
</p>

<hr>

<h2 style="text-align: center; color: green;">Limitations</h2>
<p>
While TigerLLM demonstrates impressive performance, limitations remain. The Bangla-TextBook corpus is restricted to Grades 6–12 and may not capture broader linguistic nuances, and the Bangla-Instruct dataset covers a limited subset of instruction types. Additionally, the models are currently limited to 1B and 9B parameters due to computational constraints.
</p>

<hr>

<h2 style="text-align: center; color: green;">Ethical Considerations</h2>
<p>
Our approach emphasizes ethical practices by using open-source educational materials, ensuring cultural sensitivity via volunteer contributions, and applying rigorous filtering methods to avoid harmful biases. Users should implement further safeguards when deploying TigerLLM in sensitive applications.
</p>

<hr>

<h2 style="text-align: center; color: green;">References</h2>
<ul>
  <li>Alam, F., Chowdhury, S. A., et al. (2024). LLMs for low resource languages in multilingual settings.</li>
  <li>Bai, Y., Jones, A., et al. (2024). Claude 3.5 Sonnet Technical Report.</li>
  <li>Bhattacharjee, A., Hasan, T., et al. (2022). BanglaBERT: Language model pretraining and benchmarks for Bangla.</li>
  <li>Brown, T., Mann, B., et al. (2023). GPT-4 Technical Report.</li>
  <li>Brown, T., Mann, B., et al. (2020). Language models are few-shot learners.</li>
  <li>Chowdhery, A., Narang, S., et al. (2022). PaLM: Scaling language modeling with pathways.</li>
  <li>Corso, F., Pierri, F., et al. (2024). TikTokenizer research.</li>
  <li>Dubey, A., Jauhri, A., et al. (2024). The LLaMA 3 herd of models.</li>
  <li>Ekram, S. M. S., Rahman, A. A., et al. (2022). BanglaRQA benchmark.</li>
  <li>Gunasekar, S., et al. (2023). Textbooks are all you need.</li>
  <li>Hinton, G., Vinyals, O., &amp; Dean, J. (2015). Distilling the knowledge in a neural network.</li>
  <li>Hu, E. J., Wallis, P., et al. Lora: Low-rank adaptation of large language models.</li>
  <li>Mitra, A., Del Corro, L., et al. (2023). Orca 2: Teaching small language models how to reason.</li>
  <li>Ortiz Suárez, P. J., Romary, L., &amp; Sagot, B. Contextualized word embeddings for mid-resource languages.</li>
  <li>Raihan, N., Anastasopoulos, A., &amp; Zampieri, M. (2024). mHumanEval – A multilingual benchmark for code generation.</li>
  <li>Rony, M. R. A. H., et al. (2024). BanglaQuaD: A Bangla open-domain question answering dataset.</li>
  <li>Shafayat, S., et al. (2024). BEnQA: A benchmark for Bangla question answering and reasoning.</li>
  <li>Taori, R., Gulrajani, I., et al. (2023). Alpaca: A replicable instruction-following model.</li>
  <li>Team, G., et al. (2024). Gemma 2: Improving open language models at a practical size.</li>
  <li>Wang, Y., et al. (2023). Self-instruct: Aligning language models with self-generated instructions.</li>
  <li>Wang, Y., et al. (2024). MMLU-Pro: A robust multi-task language understanding benchmark.</li>
  <li>Yue, X., et al. (2024). Pangea: A fully open multilingual multimodal LLM for 39 languages.</li>
  <li>Zehady, A. K., et al. (2024). BongLLama: Llama for Bangla language.</li>
  <li>Zhang, Y., et al. (2023). Llama: Open and efficient foundation language models.</li>
</ul>

<hr>

<h2 style="text-align: center; color: green;">Appendix A: Bangla-Instruct Curation</h2>

<h3 style="text-align: center; color: green;">A.1 Volunteer Information</h3>
<p>
Seed tasks were created by <span style="color: red;">50 volunteers</span> from various Bangladeshi universities:
<ul>
  <li>15 from Computer Science and Engineering</li>
  <li>10 from Bengali Literature</li>
  <li>10 from Business Administration</li>
  <li>8 from Science and Engineering</li>
  <li>7 from Social Sciences</li>
</ul>
Each volunteer contributed 10 diverse instructions, resulting in 500 seed tasks.
</p>

<h3 style="text-align: center; color: green;">A.2 The Seed Dataset</h3>
<p>
The seed dataset covers 10 categories:
<ol>
  <li><span style="color:red;">Cultural Knowledge and Heritage</span></li>
  <li><span style="color:red;">Academic Writing</span></li>
  <li><span style="color:red;">Mathematical Problem Solving</span></li>
  <li><span style="color:red;">Programming and Technical</span></li>
  <li><span style="color:red;">Creative Writing</span></li>
  <li><span style="color:red;">Scientific Explanation</span></li>
  <li><span style="color:red;">Business and Economics</span></li>
  <li><span style="color:red;">Social Issues Analysis</span></li>
  <li><span style="color:red;">Data Analysis and Statistics</span></li>
  <li><span style="color:red;">Language and Translation</span></li>
</ol>
Each category is represented with approximately 50 tasks.
</p>

<h3 style="text-align: center; color: green;">A.3 Filtering Methodology</h3>
<p>
Filtering is based on:
<ul>
  <li><span style="color:red;">Language Adherence</span>: High Bengali word ratio, Unicode consistency, and grammar score ≥ 0.8.</li>
  <li><span style="color:red;">Cultural Sensitivity</span>: Ensuring religious neutrality, regional inclusivity, gender balance, and political neutrality.</li>
  <li><span style="color:red;">Content Quality</span>: Minimum length, coherence between instruction and response, factual accuracy, and proper formatting.</li>
  <li><span style="color:red;">Novelty Verification</span>: Ensuring low similarity with existing tasks and sufficient lexical diversity.</li>
</ul>
A pair (i, r) is accepted only if all criteria are met.
</p>

<hr>

<h2 style="text-align: center; color: green;">Appendix B: Experimentation Details</h2>

<h3 style="text-align: center; color: green;">B.1 Experimental Setup</h3>
<p>
Pretraining was conducted on a Lambda Labs cluster with 8 NVIDIA A100 GPUs (40GB each), 512GB RAM, and 2TB storage (~120 hours with gradient checkpointing). Finetuning was performed on a single NVIDIA A100 GPU via Google Colab (~96 hours).
</p>

<h3 style="text-align: center; color: green;">B.2 Pretraining Hyperparameters (Table 3)</h3>
<table>
  <thead>
    <tr>
      <th style="color: green; text-align: center;">Hyperparameter</th>
      <th style="color: green; text-align: center;">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Per device train batch size</td>
      <td>64</td>
    </tr>
    <tr>
      <td>Gradient accumulation steps</td>
      <td>16</td>
    </tr>
    <tr>
      <td>Number of training epochs</td>
      <td>4</td>
    </tr>
    <tr>
      <td>Learning rate</td>
      <td>5×10<sup>-6</sup></td>
    </tr>
    <tr>
      <td>FP16</td>
      <td>False</td>
    </tr>
    <tr>
      <td>BF16</td>
      <td>True</td>
    </tr>
    <tr>
      <td>Dataloader num workers</td>
      <td>8</td>
    </tr>
    <tr>
      <td>Gradient checkpointing</td>
      <td>True</td>
    </tr>
    <tr>
      <td>Logging steps</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>DDP find unused parameters</td>
      <td>False</td>
    </tr>
    <tr>
      <td>Max gradient norm</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>Warmup steps</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>Evaluation strategy</td>
      <td>steps</td>
    </tr>
    <tr>
      <td>Evaluation steps</td>
      <td>1,000</td>
    </tr>
    <tr>
      <td>Save strategy</td>
      <td>steps</td>
    </tr>
    <tr>
      <td>Save steps</td>
      <td>1,000</td>
    </tr>
    <tr>
      <td>Save total limit</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Load best model at end</td>
      <td>True</td>
    </tr>
    <tr>
      <td>Metric for best model loss</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

<h3 style="text-align: center; color: green;">B.3 Finetuning Hyperparameters</h3>
<p>
Finetuning settings for TigerLLM (1B) and (9B) are detailed in Tables 4 and 5.
</p>

<table>
  <thead>
    <tr>
      <th style="color: green; text-align: center;">Parameter</th>
      <th style="color: green; text-align: center;">TigerLLM (1B)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Max Sequence Length</td>
      <td>2048</td>
    </tr>
    <tr>
      <td>Batch Size (Train/Eval)</td>
      <td>16</td>
    </tr>
    <tr>
      <td>Gradient Accumulation Steps</td>
      <td>4</td>
    </tr>
    <tr>
      <td>Number of Epochs</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Learning Rate</td>
      <td>1e-5</td>
    </tr>
    <tr>
      <td>Weight Decay</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>Warmup Steps</td>
      <td>10%</td>
    </tr>
    <tr>
      <td>Optimizer</td>
      <td>AdamW (8-bit)</td>
    </tr>
    <tr>
      <td>LR Scheduler</td>
      <td>Cosine</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>BF16</td>
    </tr>
    <tr>
      <td>Evaluation Steps</td>
      <td>50</td>
    </tr>
    <tr>
      <td>Seed</td>
      <td>42</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th style="color: green; text-align: center;">Parameter</th>
      <th style="color: green; text-align: center;">TigerLLM (9B)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Max Sequence Length</td>
      <td>2048</td>
    </tr>
    <tr>
      <td>Batch Size (Train/Eval)</td>
      <td>32</td>
    </tr>
    <tr>
      <td>Gradient Accumulation Steps</td>
      <td>8</td>
    </tr>
    <tr>
      <td>Number of Epochs</td>
      <td>3</td>
    </tr>
    <tr>
      <td>Learning Rate</td>
      <td>1e-6</td>
    </tr>
    <tr>
      <td>Weight Decay</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Warmup Steps</td>
      <td>15%</td>
    </tr>
    <tr>
      <td>Optimizer</td>
      <td>AdamW (8-bit)</td>
    </tr>
    <tr>
      <td>LR Scheduler</td>
      <td>Cosine</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>BF16</td>
    </tr>
    <tr>
      <td>Evaluation Steps</td>
      <td>250</td>
    </tr>
    <tr>
      <td>Seed</td>
      <td>42</td>
    </tr>
  </tbody>
</table>

<hr>

<h2 style="text-align: center; color: green;">Appendix C: TigerLLM - Training Pipeline</h2>
<p>
Figure 2 illustrates the multi-stage training pipeline for producing both TigerLLM (1B) and TigerLLM (9B). The process begins with pre-trained models (LLaMA 3.2 and Gemma-2), followed by continual pretraining on the Bangla-TextBook corpus and subsequent finetuning on the Bangla-Instruct dataset. Figures 3 and 4 depict the loss curves during the pretraining and finetuning stages respectively.
</p>
