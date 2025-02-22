import torch
import os
# ...existing code...
from moba_research.preprocessing import ContentProcessor
from moba_research.cache import KVCache
from moba_research.attention import moba_attn_varlen, MoBAConfig  # Assumed available from MoBA module

# --- Configuration ---
CACHE_NAME = "moba_kv_cache"  # Saved as kv_cache/{CACHE_NAME}.pt
MOBA_CHUNK_SIZE = 64
MOBA_TOPK = 4
moba_config = MoBAConfig(moba_chunk_size=MOBA_CHUNK_SIZE, moba_topk=MOBA_TOPK)
kv_cache = KVCache("~/workspace/moba_poc/kv_cache")
processor = ContentProcessor()

def load_articles():
	# In reality, load from files/database; here we use dummy articles.
	return [
		"Article 1 text content about AI and LLMs.",
		"Article 2 text discussing MoBA in detail.",
		"Another article on efficient attention mechanisms."
	]

def create_and_save_kv_cache():
	articles = load_articles()
	combined_k_list = []
	combined_v_list = []
	for i, article in enumerate(articles):
		print(f"Preprocessing article {i+1}/{len(articles)}...")
		k, v = processor.preprocess_article(article)
		combined_k_list.append(k)
		combined_v_list.append(v)
	# Combine KV data from all articles.
	combined_k = torch.cat(combined_k_list, dim=0)
	combined_v = torch.cat(combined_v_list, dim=0)
	cache_data = {'k': combined_k, 'v': combined_v}
	kv_cache.save_kv_cache_to_ssd(cache_data, CACHE_NAME)

def answer_question_with_cache(question_text, moba_config):
	cache_data = kv_cache.load_kv_cache_from_ssd(CACHE_NAME)
	if cache_data is None:
		return "KV Cache not loaded. Cannot answer question."
	# Ensure question tensor and cached KV are in fp16
	question_tensor = processor.encode_question(question_text).cuda().half()
	device = question_tensor.device
	cached_k = cache_data['k'].to(device).cuda().half()
	cached_v = cache_data['v'].to(device).cuda().half()
	# Prepare cumulative sequence lengths.
	seqlen_q = question_tensor.shape[0]
	seqlen_kv = cached_k.shape[0]
	cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
	cu_seqlens_kv = torch.tensor([0, seqlen_kv], dtype=torch.int32, device=device)
	# Call MoBA Attention (moba_attn_varlen returns a tensor output).
	attn_output = moba_attn_varlen(
		question_tensor, cached_k, cached_v,
		cu_seqlens_kv, seqlen_kv,
		moba_config.moba_chunk_size, moba_config.moba_topk
	)
	# In a full model, further decoding would occur; we return a placeholder answer.
	return "Answer generated based on MoBA attention and KV Cache. (Placeholder Answer)"

if __name__ == "__main__":
	# 1. Create and save the KV Cache.
	create_and_save_kv_cache()
	# 2. Load the cache and answer questions.
	question1 = "What is MoBA and how does it improve LLMs?"
	answer1 = answer_question_with_cache(question1, moba_config)
	print(f"Question: {question1}\nAnswer: {answer1}\n")
	question2 = "What are some applications of MoBA?"
	answer2 = answer_question_with_cache(question2, moba_config)
	print(f"Question: {question2}\nAnswer: {answer2}\n")
