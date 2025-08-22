# Frequently Asked Questions (FAQ)

## Document Handling

<details>
<summary><strong>What document formats are supported?</strong></summary>

The default setup works well with plain text (`.txt`) and PDF (`.pdf`). Other formats (e.g., `.docx`, `.md`) may work, although the performance may become worse.
</details>

<details>
<summary><strong>Where should I place my documents?</strong></summary>

Place your documents in the `data/` directory.
</details>

<details>
<summary><strong>How large can the documents be?</strong></summary>

There’s no strict size limit, but very large documents may reduce performance or exceed the model's context window. For best results, split long documents into smaller chunks.
</details>

## Behavior

<details>
<summary><strong>Why is the answer incorrect or omits relevant information?</strong></summary>

This may happen if the LLM is too small. Larger models are better at synthesizing information from multiple documents. Increasing the number of retrieved passages (*top n*) can also help improve answer quality.
</details>

<details>
<summary><strong>Why are no relevant documents being retrieved?</strong></summary>

Try lowering the relevance threshold in the interface. A high threshold can filter out potentially useful passages.
</details>

<details>
<summary><strong>How can I produce an answer in a specific format?</strong></summary>

You can customize the answer style by editing the prompt templates in `./ragsst/ragtool.py`, inside the functions `get_context_prompt()` and `get_condenser_prompt()`.
</details>

## Troubleshooting

<details>
<summary><strong>I started the tool, but nothing happens in the browser. What should I check?</strong></summary>

1. Confirm that your virtual environment is active.  
2. Look for the local URL printed in the terminal (e.g., `http://127.0.0.1:7860`) and open it manually.  
3. Ensure that no firewall or process is blocking the port.  
4. Try restarting the app or changing the port if needed.
</details>

<details>
<summary><strong>Ollama throws an error about memory or model size.</strong></summary>

Use a smaller model such as `llama3.2:1b` or `qwen2.5:3b`, especially on systems without a dedicated GPU.
</details>

---

If your question is not answered here, feel free to open an issue on the repository or contact the maintainer listed in the README.
