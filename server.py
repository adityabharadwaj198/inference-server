# Re-wrote this in server.ipynb 

import json 
import time 
from typing import Optional 
import uvicorn 

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM 


# this is required for the model to run on the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available")

"""
This function is used to synchronize the GPU.
It is required to ensure that the GPU is properly synchronized with the CPU.
Otherwise, the model will not be able to run on the GPU.
"""
def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

start_time = time.time()
Model_id = "Qwen/Qwen3-4B"

# for loading the tokenizer specific to this model. We also do this in our embedding gen pipeline at Uber. This exists inside the model class
tokenizer = AutoTokenizer.from_pretrained(Model_id)


model = AutoModelForCausalLM.from_pretrained(Model_id, dtype=torch.float16, device_map="auto").eval()

print(f"Loaded model {Model_id} in {time.time() - start_time:.2f} seconds")


def prepare_inputs(prompt: str):
    # TODO 1: create system and user messages, then render the generation prompt
    messages = [{"role": "system", "content": "You are a good assistant. Keep your responses short and concise. Do not overthink your responses. Do not make up facts. Do not hallucinate. Ground your responses in reality."}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict = True)
    
    inputs = inputs.to(model.device)

    assert messages, "Add the prompt messages"
    assert inputs is not None, "Render the prompt with the tokenizer"
    print(inputs)
    return inputs   

def select_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # TODO 2: scale logits, apply the top-k cutoff, and sample one token.
    scaled_logits = logits / temperature 
    if top_k is not None:
        top_values = torch.topk(scaled_logits, top_k, dim=-1).values
        # keep the trailing dim so this broadcasts over the vocab, and use
        # masked_fill so the -inf stays on the model's device
        cutoff = top_values[:, -1:]
        scaled_logits = scaled_logits.masked_fill(scaled_logits < cutoff, float("-inf"))

    probabilities = scaled_logits.softmax(dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)
    assert next_token is not None, "Sample one token"
    return next_token


@torch.inference_mode()
def generate_cached(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> dict:
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"] # contains all the token ids for the input prompt in the order they appear
    print(f"generated_ids: {generated_ids}") 
    attention_mask = inputs["attention_mask"] # contains a binary mask of the same length as the input prompt, with 1s for all the tokens that are not padding tokens.
    prompt_length = generated_ids.shape[1] # length of the input prompt after tokenization
    first_token_at = None

    sync_gpu() # synchronize the GPU - this is required to ensure that the GPU is properly synchronized with the CPU. Otherwise, the model will not be able to run on the GPU.
    started_at = time.perf_counter()
    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )


    past_key_values = outputs.past_key_values 

    for _ in range(max_new_tokens):
        """
        What is inside outputs? 
        The outputs object is an instance of CausalLMOutputWithPast. 
        Because you called the model directly, it contains:
        
        logits: A massive tensor of shape (batch_size, sequence_length, vocab_size). 
        This contains the unnormalized prediction scores for every single token position in your input. 
        If you want to know what the model thinks the next token should be, 
        you look at the very last position of these logits.
        
        loss: This will be None because you didn't pass a labels argument to the model.
        
        past_key_values: This will be None because you set use_cache=False.
        """

        # TODO 4: select the next token and reuse past_key_values for the next model call.
        next_token = select_next_token(
            outputs.logits[:, -1, :], # passing in the last token's logits. The last token's logits contain raw scores for the entire vocabulary
            temperature,
            top_k,
        )

        if next_token.item() == tokenizer.eos_token_id:
            break
        
        if first_token_at is None: # this is used to calculate the time it takes to generate the first token
            sync_gpu()
            first_token_at = time.perf_counter()


        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )

        past_key_values = outputs.past_key_values 
        

    sync_gpu()
    finished_at = time.perf_counter()
    completion_ids = generated_ids[0, prompt_length:]
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "token_ids": completion_ids.tolist(),
        "input_tokens": prompt_length,
        "ttft_ms": (first_token_at - started_at) * 1000,
        "latency_ms": (finished_at - started_at) * 1000,
    }

@torch.inference_mode()
def generate_stream(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
):
    """Yield SSE chunks as tokens are produced. FastAPI's StreamingResponse
    iterates this generator and flushes each yield to the client immediately.
    """
    if top_k is not None and top_k <= 0:
        top_k = None

    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]
    previous_text = ""

    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )


    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_token = select_next_token(
            outputs.logits[:, -1, :],
            temperature,
            top_k,
        )
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)],
            dim=1,
        )

        # Decode the whole completion so far, then emit only the new suffix.
        # Needed because one token is not always a complete UTF-8 / BPE piece.
        current_text = tokenizer.decode(
            generated_ids[0, prompt_length:],
            skip_special_tokens=True,
        )
        if not current_text.endswith("\ufffd"):
            chunk = current_text[len(previous_text):]
            previous_text = current_text
            if chunk:
                yield f"data: {json.dumps({'text': chunk})}\n\n"

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

    yield "data: [DONE]\n\n"


from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    # TODO 7: add validated prompt, token limit, temperature, and top-k fields.
    prompt: str = Field(..., description="The prompt to generate from")
    token_limit: int = Field(..., description="The maximum number of tokens to generate")
    temperature: float = Field(..., description="The temperature to use for generation")
    top_k: int = Field(..., description="The top-k value to use for generation")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = kwargs.get("prompt", "")
        self.token_limit = kwargs.get("token_limit", 32)
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_k = kwargs.get("top_k", None)

class GenerateResponse(BaseModel):
    # TODO 7: expose text, timing, token count, throughput, and cache usage.
    text: str = Field(..., description="The generated text")
    timing: dict = Field(..., description="The timing information")
    token_count: int = Field(..., description="The number of tokens in the generated text")
    throughput: float = Field(..., description="The throughput of the generation")
    cache_usage: str = Field(..., description="The cache usage information")
    status_code: int = Field(..., description="The status code of the response")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = kwargs.get("text", "")
        self.timing = kwargs.get("timing", {})
        self.token_count = kwargs.get("token_count", 0)
        self.throughput = kwargs.get("throughput", 0.0)
        self.cache_usage = kwargs.get("cache_usage", "")
    


from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(title="tinyinference")

@app.get("/health")
def health():
    return {"status": "ok"}

# TODO 8: implement the measured generation and streaming endpoints.
@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    response = generate_cached(
        request.prompt,
        request.token_limit,
        request.temperature,
        request.top_k,
    )

    return GenerateResponse(
        text=response["text"],
        timing={"latency_ms": response["latency_ms"], "ttft_ms": response["ttft_ms"]},
        token_count=response["input_tokens"],
        throughput=response.get("throughput", 0.0),
        cache_usage="true", # because we're calling generate_cached here 
        status_code=200,
    )

@app.post("/stream")
async def stream_text(request: GenerateRequest):
    # generate_stream() returns a generator. Do not list() or join() it here —
    # StreamingResponse iterates it and sends each yield as an SSE frame.
    return StreamingResponse(
        generate_stream(
            request.prompt,
            request.token_limit,
            request.temperature,
            request.top_k,
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
