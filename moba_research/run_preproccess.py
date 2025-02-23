import asyncio
import torch
import os

from moba_research.llm_provider import OllamaProvider
from moba_research.preprocessing import ContentProcessor
from moba_research.cache import KVCache
from moba_research.attention import moba_attn_varlen, MoBAConfig  # Assumed available from MoBA module
from moba_research.preprocessing.ollama_embedding_preprocessor import OllamaEmbeddingPreprocessor

# --- Configuration ---
CACHE_NAME = "moba_kv_cache"  # Saved as kv_cache/{CACHE_NAME}.pt
MOBA_CHUNK_SIZE = 64
MOBA_TOPK = 4
moba_config = MoBAConfig(moba_chunk_size=MOBA_CHUNK_SIZE, moba_topk=MOBA_TOPK)
kv_cache = KVCache("/home/gsegura/workspace/moba_poc/kv_cache")
llm_provider = OllamaProvider(host="http://ai-server:11434") # Example provider for Ollama embeddings
processor = OllamaEmbeddingPreprocessor(llm_provider)

def load_articles():
	# In reality, load from files/database; here we use dummy articles.
	# For demonstration, articles starting with "http" will be treated as URLs.
	return [
		"https://tridao.me/publications/flash2/flash2.pdf", 
		"Article 2 text discussing MoBA in detail.",
		"Another article on efficient attention mechanisms."
	]

async def create_and_save_kv_cache():
	articles = load_articles()
	combined_k_list = []
	combined_v_list = []
	for i, article in enumerate(articles):
		print(f"Preprocessing article {i+1}/{len(articles)}...")
		# If article is a URL, fetch markdown content; otherwise use as plain text.
		if article.startswith("http"):
			article_content = await processor.get_markdown_content_from_url(article)
		else:
			article_content = article
		# Preprocess the article content to get K and V tensors.
		k, v = await processor.preprocess_article(article_content)
		combined_k_list.append(k)
		combined_v_list.append(v)
	# Combine KV data from all articles.
	combined_k = torch.cat(combined_k_list, dim=0)
	combined_v = torch.cat(combined_v_list, dim=0)
	cache_data = {'k': combined_k, 'v': combined_v}
	kv_cache.save_kv_cache_to_ssd(cache_data, CACHE_NAME)

async def answer_question_with_cache_old(question_text, moba_config):
	# Load KV Cache from SSD
	cache_data = kv_cache.load_kv_cache_from_ssd(CACHE_NAME)
	if cache_data is None:
		return "KV Cache not loaded. Cannot answer question."
	# Encode question to get Q tensor and ensure tensor is on GPU with fp16 precision
	question_tensor = (await processor.encode_question(question_text)).cuda().half()
	device = question_tensor.device
	# Retrieve cached K,V, move to device and convert to fp16
	cached_k = cache_data['k'].to(device).half()
	cached_v = cache_data['v'].to(device).half()
	# Prepare cumulative sequence lengths and max_seqlen based on cached KV dimensions
	seqlen_kv = cached_k.shape[0]
	cu_seqlens = torch.tensor([0, seqlen_kv], dtype=torch.int32, device=device)
	max_seqlen = seqlen_kv
	# Call MoBA Attention: moba_attn_varlen expects (q, k, v, cu_seqlens, max_seqlen, moba_chunk_size, moba_topk)
	attn_output = moba_attn_varlen(
		question_tensor, cached_k, cached_v,
		cu_seqlens, max_seqlen,
		moba_config.moba_chunk_size, moba_config.moba_topk
	)
	# For now, return a placeholder answer. In a full model, attn_output would be further decoded.
	return "Answer generated based on MoBA attention and KV Cache. (Placeholder Answer)"

async def answer_question_with_cache(question_text, moba_config, ollama_preprocessor): # Pass preprocessor instance
	cached_kv_data = kv_cache.load_kv_cache_from_ssd(CACHE_NAME)
	if cached_kv_data is None:
		return "KV Cache not loaded. Cannot answer question."

	question_tensor_q = await ollama_preprocessor.encode_question(question_text) # Use async encode_question
	cached_k = cached_kv_data['k'].to(question_tensor_q.device).half()
	cached_v = cached_kv_data['v'].to(question_tensor_q.device).half()

	# --- Prepare cu_seqlens and max_seqlen --- (No change needed here)
	seqlen_q = question_tensor_q.shape[0]
	seqlen_kv = cached_k.shape[0]
	cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=question_tensor_q.device)
	cu_seqlens_kv = torch.tensor([0, seqlen_kv], dtype=torch.int32, device=question_tensor_q.device)
	max_seqlen_q = seqlen_q
	max_seqlen_kv = seqlen_kv

	# --- Call MoBA Attention --- (No change needed here)
	attn_output = moba_attn_varlen(
		question_tensor_q, cached_k, cached_v, cu_seqlens_kv, max_seqlen_kv,
		moba_config.moba_chunk_size, moba_config.moba_topk
	)

	return "Answer generated based on MoBA attention and KV Cache. (Placeholder Answer with Ollama Embeddings)" # Updated message

if __name__ == "__main__":
	async def main():
		# 1. Create and save the KV Cache.
		await create_and_save_kv_cache()
		# 2. Load the cache and answer questions.
		question1 = "What is MoBA and how does it improve LLMs?"
		answer1 = await answer_question_with_cache(question1, moba_config)
		print(f"Question: {question1}\nAnswer: {answer1}\n")
		question2 = "What are some applications of MoBA?"
		answer2 = await answer_question_with_cache(question2, moba_config)
		print(f"Question: {question2}\nAnswer: {answer2}\n")

	asyncio.run(main())
