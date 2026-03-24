# FAQ

## `Leiden.EmptyNetworkError: EmptyNetworkError`

This happens when `nano-graphrag` tries to compute communities on an empty graph. In practice, the usual cause is that the LLM failed to extract entities or relationships from the source text.

Things to check:

- Verify the LLM is returning the expected extraction format
- Try a stronger model if the current one is too weak for extraction
- Inspect the raw model response before community generation starts

Expected extraction shape:

```text
("entity"<|>"Cruz"<|>"person"<|>"Cruz is associated with a vision of control and order, influencing the dynamics among other characters.")
```

If your model does not reliably follow that format, adding a stronger system instruction can help:

```json
{
  "role": "system",
  "content": "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format shown in the provided example."
}
```

## `Processed 42 chunks, 0 entities, 0 relations`

If you see warnings like:

```text
Didn't extract any entities
No new entities found
```

and you are using Ollama, the model context window is often the problem. A default `num_ctx` of `2048` is usually too small for the extraction prompt.

One fix is to create a model variant with a larger context window:

```bash
ollama show --modelfile qwen2 > Modelfile
```

Add this line under `FROM`:

```text
PARAMETER num_ctx 32000
```

Then create the adjusted model:

```bash
ollama create -f Modelfile qwen2:ctx32k
```

After that, switch your config to `qwen2:ctx32k`.
