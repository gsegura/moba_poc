# Deep Research powered by MoBA and KV Cache - Implementation Plan (Draft v1)

**Project Goal:** Create a sample implementation of a "Deep Research" agent that leverages MoBA (Mixture of Block Attention) and a custom KV Cache stored on SSD to generate comprehensive reports based on web content.

**Phase 1: Core MoBA + Page-Level KV Cache + Search Integration (Initial Implementation Focus)**

## 1. Core Components:

*   **MoBA Attention Mechanism:**
    *   Utilize the provided PyTorch/FlashAttention MoBA code (`moba_attn_varlen`, `MixedAttention`, etc.).
    *   Define `MoBAConfig` (for `moba_chunk_size`, `moba_topk`).
    *   Keep the initial MoBA implementation as is for now.

*   **Page-Level KV Cache (SSD Storage):**
    *   **Cache Storage Format:** `torch.save` to SSD.
    *   **Cache Structure:** Dictionary in Python. Key: URL (or hash of URL), Value: Tuple of (K, V) tensors for the entire webpage.
    *   Implement `save_kv_cache_to_ssd(cache_data, cache_path)` and `load_kv_cache_from_ssd(cache_path)` functions.
    *   Define `CACHE_PATH` constant for SSD cache file location.

*   **Search Integration:**
    *   Use **Google Custom Search API** for initial URL retrieval.
    *   API Key management and query formulation will be needed.
    *   Function to perform search and retrieve list of relevant URLs based on user query.

*   **Webpage Content Extraction & Markdown Conversion:**
    *   Fetch HTML content from URLs using `requests` or similar.
    *   Convert HTML to Markdown using `html2text`.
    *   Implement basic cleaning (removal of boilerplate, ads, navigation – can be improved later).
    *   Function `get_markdown_content_from_url(url)` to handle fetching, conversion, and cleaning.

*   **Preprocessing & Embedding:**
    *   **Simplified Tokenization:** Basic word splitting for now. (Replace with a proper tokenizer later).
    *   **Dummy Embedding:** Use `torch.randn` to generate placeholder K, V, and Q tensors of appropriate dimensions. (Replace with a real embedding model later, aligned with a chosen LLM architecture if we expand further).
    *   `preprocess_article(article_text)` function to generate (K, V) for a given article (Markdown text).
    *   `encode_question(question_text)` function to generate Q for a question (Markdown text).

*   **Report Generation (Simplified):**
    *   For Phase 1, focus on getting MoBA attention output.
    *   Initial report generation can be a placeholder like: "Summary of relevant information based on MoBA attention... (Placeholder Output)".  We'll improve this in later phases.

## 2. Implementation Steps (Phase 1):

1.  **Set up PyTorch and FlashAttention environment.**
2.  **Implement KV Cache functions:** `save_kv_cache_to_ssd`, `load_kv_cache_from_ssd`.
3.  **Implement Search Integration:** Function to use Google Custom Search API and retrieve URLs.
4.  **Implement Webpage Content Extraction and Markdown Conversion:** `get_markdown_content_from_url`.
5.  **Implement Simplified Preprocessing & Embedding:** `preprocess_article`, `encode_question` (using dummy embeddings initially).
6.  **Implement `create_and_save_kv_cache()` function:**
    *   Loads articles (for now, use dummy `load_articles()` returning sample text).
    *   For each article, get Markdown content, preprocess to get (K, V), and store in page-level cache.
    *   Save the cache to SSD.
7.  **Implement `answer_question_with_cache(question_text, moba_config)` function:**
    *   Load KV cache from SSD.
    *   Encode question to get Q.
    *   Retrieve cached (K, V) (for *all* pages initially - we'll refine relevance later).
    *   Prepare `cu_seqlens`, `max_seqlen` for MoBA.
    *   Call `moba_attn_varlen`.
    *   Return placeholder report output.
8.  **Create `main` function for example usage:**
    *   Option to create and save cache.
    *   Example questions and calls to `answer_question_with_cache` to test the flow.

## 3. Technology Stack (Phase 1):

*   **Programming Language:** Python
*   **Deep Learning Framework:** PyTorch
*   **Efficient Attention:** FlashAttention
*   **Markdown Processing:** `html2text` (or similar), `markdown` (Python-Markdown) if needed for further Markdown parsing.
*   **Web Scraping/Fetching:** `requests`
*   **Search API:** Google Custom Search API (or similar)
*   **Storage:** SSD for KV Cache

## 4.  Next Steps (Beyond Phase 1 - Iterations & Improvements):

*   **Phase 2: Block-Level KV Cache:** Implement block segmentation of webpages (HTML structure-based or content-based) and modify cache to store (K, V) per block. Update `preprocess_article`, cache saving/loading, and MoBA attention to work with block-level cache.
*   **Phase 3: Improved Report Generation:** Replace placeholder report output with more sophisticated decoding and report synthesis logic. Potentially integrate with a pre-trained LLM decoder or summarization module.
*   **Phase 4: Relevance Refinement:**  Explore techniques for pre-filtering webpages or blocks for better relevance, and refine the MoBA gating mechanism for more accurate block selection.
*   **Phase 5: Performance Optimization:** Optimize SSD access, GPU utilization, and overall inference speed. Investigate more efficient Markdown processing if it becomes a bottleneck.
*   **Phase 6: User Interface & Agent Features:**  Develop a user interface and add more agent-like features (query refinement, iterative research, handling different question types, etc.).

**Validation and Feasibility:**

*   The architecture is feasible using the proposed components.
*   Starting with Page-Level KV Cache and simplified preprocessing makes Phase 1 achievable as a sample implementation.
*   The iterative approach allows for gradual enhancement and addresses complexity in stages.

**Doubts/Clarifications Addressed:**

*   **Search Strategy:**  Chosen Search API for initial simplicity.
*   **Cache Granularity:** Starting with Page-Level, iterating to Block-Level.
*   **Relevance Pre-filtering:** Deferred for later phases, relying on MoBA's gating initially.
*   **Report Generation Process:** Simplified to a single MoBA pass and placeholder output for Phase 1.
*   **Markdown Processing:**  `html2text` + basic tokenization pipeline proposed.

This markdown provides a structured plan for the initial implementation.  Let me know if this direction makes sense and if you'd like to proceed with creating the implementation based on this plan. We can then refine it further as we progress.



# Implementation Details

**1. Framework Selection: PyTorch and FlashAttention**

Given the provided code snippet and the performance requirements for MoBA (efficient long-context attention), **PyTorch** is a natural choice as the deep learning framework.  The provided code already leverages **FlashAttention**, which is crucial for efficient attention computations, especially with long sequences.

Therefore, we will proceed with **PyTorch** and **FlashAttention** for our implementation. This aligns well with the existing MoBA code and the need for speed.

**2. Understanding the MoBA Paper and Sample Code**

Let's dissect the MoBA concept and the provided Python code:

**2.1. MoBA Paper Idea - Key Concepts:**

* **Long Context Challenge:**  Traditional attention in LLMs has quadratic complexity with sequence length, making long contexts computationally expensive.
* **Mixture of Block Attention (MoBA):** MoBA addresses this by dividing the input sequence into blocks and selectively attending to only the *most relevant* blocks for each query token.
* **Block Division:** The input sequence is partitioned into smaller, consecutive token blocks.
* **Gating Mechanism (Inspired by MoE):** A gating mechanism dynamically selects which blocks are most informative for each query token. This is done by having each query token attend to representative "keys" from each block and determining relevance based on attention scores.
* **Efficiency:** By attending to a subset of blocks instead of the entire sequence, MoBA reduces computational cost, enabling longer context windows.
* **Causality Preservation:** MoBA ensures causality (no "looking ahead") with two rules:
    * **No Future Blocks:** Query tokens cannot attend to blocks that appear later in the sequence.
    * **Current Block Focus:** Query tokens always attend to their "current block" to maintain immediate context.

**2.2. Deconstructing the Sample Code:**

The provided code gives us a good starting point. Let's analyze the key components:

* **`calc_chunks(cu_seqlen, moba_chunk_size)`:**
    * **Purpose:**  This function is crucial for dividing the input sequences into chunks and preparing metadata for MoBA.
    * **`cu_seqlen` (Cumulative Sequence Lengths):**  Standard FlashAttention input, indicating the start and end indices of sequences within a batched input.
    * **`moba_chunk_size`:**  The size of each block/chunk.
    * **Functionality:**
        * Calculates the number of chunks per batch and across the entire input.
        * Determines chunk sizes, handling cases where batches are not perfectly divisible by `moba_chunk_size`.
        * `filtered_chunk_indices`:  Identifies chunks that are *not* the last chunk of each batch (these are the chunks targeted for MoBA attention; the last chunk is for self-attention).
        * Returns metadata like `cu_chunk` (cumulative chunk start indices), `filtered_chunk_indices`, `chunk_to_batch` (chunk to batch mapping), etc., which are used in subsequent MoBA attention calculations.

* **`MixedAttention.forward(ctx, q, k, v, ...)` and `MixedAttention.backward(ctx, d_output)`:**
    * **Purpose:** This is a PyTorch `autograd.Function` that implements the core logic of mixing self-attention and MoBA attention within the forward and backward passes.
    * **Inputs:**
        * `q, k, v`:  Query, Key, Value tensors (in FlashAttention format).
        * `self_attn_cu_seqlen`: Cumulative sequence lengths for self-attention (in MoBA, this seems to be based on chunk boundaries).
        * `moba_q, moba_kv, moba_cu_seqlen_q, moba_cu_seqlen_kv`:  Data specifically prepared for MoBA attention, derived from the chunks and gating mechanism.
        * `moba_chunk_size`, `max_seqlen`, `moba_q_sh_indices`:  Configuration and indexing information.
    * **Functionality (Forward):**
        1. **Self-Attention:** Performs standard FlashAttention (`_flash_attn_varlen_forward`) on the entire input (`q, k, v`).
        2. **MoBA Attention:** Performs FlashAttention on the *selected* MoBA queries (`moba_q`) and corresponding keys and values (`moba_kv`).  Crucially, `moba_kv` and `moba_cu_seqlen_kv` are designed to represent the selected blocks/chunks.
        3. **Mixing Self and MoBA Attention:**  Combines the outputs of self-attention and MoBA attention using an online softmax-like approach based on log-sum-exp (LSE). This ensures proper weighting of the two attention mechanisms.
    * **Functionality (Backward):** Implements the gradient computation for the mixed attention mechanism, using `_flash_attn_varlen_backward`.

* **`moba_attn_varlen(q, k, v, cu_seqlens, max_seqlen, moba_chunk_size, moba_topk)`:**
    * **Purpose:** This is the high-level function that orchestrates the entire MoBA attention process. It takes the standard Q, K, V, `cu_seqlens`, and MoBA configurations as input.
    * **Inputs:**
        * `q, k, v`: Query, Key, Value tensors (input to the attention layer).
        * `cu_seqlens`, `max_seqlen`: FlashAttention sequence length parameters.
        * `moba_chunk_size`, `moba_topk`: MoBA-specific hyperparameters (`moba_chunk_size` for block size, `moba_topk` for the number of top blocks to attend to).
    * **Functionality:**
        1. **Chunk Preparation:** Calls `calc_chunks` to divide the input into chunks and get metadata.
        2. **Early Exit (No MoBA):** If `moba_topk` is small enough (or no filtered chunks), it may decide to just perform standard self-attention (FlashAttention) directly if MoBA attention is deemed unnecessary.
        3. **Key Gate Weight Calculation & Gating:**
            * Calculates `key_gate_weight`:  Represents each filtered chunk by averaging the key vectors within that chunk. This acts as a "summary" of each block for the gating mechanism.
            * Computes `gate` logits:  Measures the relevance of each chunk to each query token by attending (via dot product) to the `key_gate_weight`.
            * Applies masking (`gate_inf_mask`) to ensure causality (no attending to future blocks or tokens beyond the current sequence in a batch).
        4. **Top-K Block Selection:** Selects the `moba_topk` most relevant blocks based on the `gate` logits using `torch.topk`.
        5. **Preparation of MoBA Q, KV, and `cu_seqlens`:**
            * Identifies query tokens (`moba_q_indices`) that should attend to the selected MoBA blocks based on the top-k gating results.
            * Creates `moba_q` (the selected query tokens) and `moba_kv` (the keys and values from the selected blocks).
            * Constructs `moba_cu_seqlen_q` and `moba_cu_seqlen_kv` – cumulative sequence lengths specifically for the MoBA attention part, reflecting the selected queries and blocks.
        6. **Calls `MixedAttention.apply`:**  Executes the core mixed self-attention and MoBA attention using the prepared data and metadata.
        7. **Returns `attn_output`:** The final output of the MoBA attention mechanism.

* **`MoBAConfig` Dataclass:**  Simple configuration dataclass to hold `moba_chunk_size` and `moba_topk`.
* **`moba_layer` function:**  A higher-level function to integrate `moba_attn_varlen` into a layer within a model, handling input format conversions (HF to FA and back) and calling `flash_attn_func` for the decoding phase (when query length is not equal to key length).

**3. Iterating on the Concept: Adding a Custom KV Cache**

Now, let's focus on integrating a custom KV cache that stores the cache on SSD.

**3.1. Identifying What to Cache:**

In the MoBA mechanism, the **Keys (K) and Values (V) associated with the *blocks*** are the primary candidates for caching.  The idea is that if we have a corpus of articles/content, we can pre-process them, divide them into blocks, and then cache the K and V representations of these blocks.

When we want to ask questions "about" this corpus, we can load the cached K and V and use them in the MoBA attention mechanism.  The Query (Q) will still be dynamically generated based on the user's question.

**3.2. Custom KV Cache Design and SSD Storage:**

* **Cache Structure:** We can use a dictionary to store our KV cache.  The keys of the dictionary could be identifiers for the articles or content blocks, and the values would be tuples of (K, V) tensors for those blocks.  For simplicity in this initial example, we might just cache a single combined K and V for the entire corpus, or a few articles, rather than block-level caching directly on SSD (we can refine this later).
* **Storage Format:**
    * **`torch.save`:**  For simplicity and direct PyTorch tensor compatibility, `torch.save` is a good starting point. It allows us to serialize Python objects (like our dictionary containing tensors) to disk.
    * **SSD Path:** We need to define a path on the SSD where we will save the cache file.  This can be configurable.
* **Simplified Cache (Initial Implementation):** For our *first* sample, let's make it simpler. Instead of caching *per block* directly on SSD which could be complex to manage at first, let's cache the entire pre-computed `k` and `v` tensors *after* processing the articles.

**3.3. Implementation Steps for KV Cache Integration:**

**Step 3.3.1: Data Preprocessing and Cache Creation (Stored on SSD)**

1.  **Load Articles/Content:** We'll need a way to load our articles or content.  For a sample, we can use dummy text or load from simple text files. Let's assume we have a function `load_articles()` that returns a list of text articles.

2.  **Preprocessing and MoBA Input Preparation (Simplified):**
    *   For this initial example, let's **simplify** the preprocessing and assume we convert each article into a sequence of tokens.  We'll simulate the process of creating K and V tensors.  In a real application, this would involve embedding models and more complex tokenization.
    *   Let's create a placeholder function `preprocess_article(article_text)` that, for now, just converts the text to a tensor of random embeddings representing `k` and `v`.  In a real scenario, you would use an actual embedding model based on your LLM. This function will return `k` and `v` tensors.

3.  **Cache Saving to SSD:**
    *   Create a function `save_kv_cache_to_ssd(cache_data, cache_path)`:
        *   Takes `cache_data` (which will be a dictionary containing our K and V tensors) and `cache_path` (SSD file path) as input.
        *   Uses `torch.save(cache_data, cache_path)` to save the data to the SSD.

**Step 3.3.2: Querying with KV Cache (Loading from SSD)**

1.  **Cache Loading from SSD:**
    *   Create a function `load_kv_cache_from_ssd(cache_path)`:
        *   Takes `cache_path` as input.
        *   Uses `torch.load(cache_path, map_location=torch.device('cpu'))` to load the cache from the SSD. We load to CPU first to avoid immediately filling GPU memory, and can move to GPU later if needed.
        *   Returns the loaded `cache_data` dictionary.

2.  **Question Encoding (Query Q):**
    *   When a question is asked, we need to encode it to create the Query tensor (Q).  Similar to article preprocessing, let's create a simplified function `encode_question(question_text)` that returns a Q tensor of random embeddings for now. In reality, you'd use the same embedding model as for the articles.

3.  **MoBA Attention with Cached KV:**
    *   Create a function `answer_question_with_cache(question_tensor_q, cached_kv_data, moba_config)`:
        *   Takes the encoded `question_tensor_q`, the `cached_kv_data` (loaded from SSD), and `moba_config` as input.
        *   Extracts the cached `k` and `v` tensors from `cached_kv_data`.
        *   It needs to construct the `cu_seqlens` and `max_seqlen` appropriately, based on the shapes of `question_tensor_q`, `k`, and `v`.
        *   Calls the `moba_attn_varlen` function (or a simplified version if needed for demonstration) using `question_tensor_q`, the cached `k`, `v`, and the appropriate sequence length parameters.
        *   Returns the `attn_output` (which would be processed further in a full LLM to generate an answer).

**4. Sample Code Structure (Conceptual):**

```python
# --- MoBA Code (from provided snippet) ---
# ... (calc_chunks, MixedAttention, moba_attn_varlen, MoBAConfig, moba_layer, etc.) ...

import torch
import os  # For SSD path management

# --- Configuration ---
CACHE_PATH = "ssd_cache/moba_kv_cache.pth" # Path on SSD to store the cache
MOBA_CHUNK_SIZE = 64
MOBA_TOPK = 4
moba_config = MoBAConfig(moba_chunk_size=MOBA_CHUNK_SIZE, moba_topk=MOBA_TOPK)

# --- Dummy Data Loading and Preprocessing ---
def load_articles():
    # In reality, load from files, database, etc.
    return [
        "Article 1 text content about AI and LLMs.",
        "Article 2 text discussing MoBA in detail.",
        "Another article on efficient attention mechanisms."
    ]

def preprocess_article(article_text):
    # Simplification: Generate random tensors for K and V
    seq_len = len(article_text.split()) # Just a placeholder for sequence length
    head_dim = 64
    num_heads = 8
    k = torch.randn(seq_len, num_heads, head_dim) # Placeholder
    v = torch.randn(seq_len, num_heads, head_dim) # Placeholder
    return k, v

def encode_question(question_text):
    # Simplification: Generate random Q tensor
    seq_len_q = len(question_text.split()) # Placeholder
    head_dim = 64
    num_heads = 8
    q = torch.randn(seq_len_q, num_heads, head_dim) # Placeholder
    return q

# --- KV Cache Functions ---
def save_kv_cache_to_ssd(cache_data, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True) # Ensure dir exists
    torch.save(cache_data, cache_path)
    print(f"KV Cache saved to: {cache_path}")

def load_kv_cache_from_ssd(cache_path):
    print(f"Loading KV Cache from: {cache_path}")
    if os.path.exists(cache_path):
        cache_data = torch.load(cache_path, map_location=torch.device('cpu')) # Load to CPU initially
        print("KV Cache loaded successfully.")
        return cache_data
    else:
        print("KV Cache file not found. Please preprocess and create the cache first.")
        return None

# --- Main Functions ---
def create_and_save_kv_cache():
    articles = load_articles()
    cached_kv_data = {} # Dictionary to store cached K and V
    combined_k_list = []
    combined_v_list = []

    for i, article in enumerate(articles):
        print(f"Preprocessing article {i+1}/{len(articles)}...")
        k, v = preprocess_article(article)
        combined_k_list.append(k)
        combined_v_list.append(v)
        # In a more advanced version, you might cache per block of the article

    # For this simplified example, let's combine all K and V from all articles
    combined_k = torch.cat(combined_k_list, dim=0)
    combined_v = torch.cat(combined_v_list, dim=0)

    cached_kv_data['k'] = combined_k
    cached_kv_data['v'] = combined_v

    save_kv_cache_to_ssd(cached_kv_data, CACHE_PATH)

def answer_question_with_cache(question_text, moba_config):
    cached_kv_data = load_kv_cache_from_ssd(CACHE_PATH)
    if cached_kv_data is None:
        return "KV Cache not loaded. Cannot answer question."

    question_tensor_q = encode_question(question_text)
    cached_k = cached_kv_data['k'].to(question_tensor_q.device) # Move to same device as Q (e.g., GPU if Q is on GPU)
    cached_v = cached_kv_data['v'].to(question_tensor_q.device)

    # --- Prepare cu_seqlens and max_seqlen ---
    seqlen_q = question_tensor_q.shape[0]
    seqlen_kv = cached_k.shape[0] # Assuming cached K and V have the same seqlen
    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=question_tensor_q.device)
    cu_seqlens_kv = torch.tensor([0, seqlen_kv], dtype=torch.int32, device=question_tensor_q.device) # We'll assume single "sequence" for cached KV for now
    max_seqlen_q = seqlen_q
    max_seqlen_kv = seqlen_kv

    # --- Call MoBA Attention ---
    attn_output = moba_attn_varlen(
        question_tensor_q, cached_k, cached_v, cu_seqlens_kv, max_seqlen_kv,
        moba_config.moba_chunk_size, moba_config.moba_topk
    )

    # --- Post-processing of attn_output to get an answer (Simplified) ---
    # In a real LLM, you'd have layers after attention to decode and generate text
    # For now, just return a placeholder
    return "Answer generated based on MoBA attention and KV Cache. (Placeholder Answer)"


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create and Save KV Cache (Run this once after preparing articles)
    create_and_save_kv_cache()

    # 2. Load Cache and Ask Questions (Run this every time you want to query)
    question1 = "What is MoBA and how does it improve LLMs?"
    answer1 = answer_question_with_cache(question1, moba_config)
    print(f"Question: {question1}\nAnswer: {answer1}\n")

    question2 = "What are some applications of MoBA?"
    answer2 = answer_question_with_cache(question2, moba_config)
    print(f"Question: {question2}\nAnswer: {answer2}\n")
```

**5. Key Improvements and Next Steps:**

* **Realistic Preprocessing:**  Replace the `preprocess_article` and `encode_question` placeholders with actual text tokenization and embedding using a pre-trained model (e.g., from Hugging Face Transformers).
* **Block-Level Caching:**  Implement caching at the *block* level as described in the MoBA paper. This would involve identifying blocks within articles and caching K and V for each block separately.  This will be more memory-efficient and potentially allow for finer-grained attention.
* **Cache Invalidation/Updates:**  In a real-world system with frequently updated articles, you'd need mechanisms to invalidate or update the KV cache when articles change.
* **Integration with Full LLM:** Integrate this MoBA implementation and KV cache into a larger LLM architecture. The `moba_layer` function is a step in this direction, but you'd need to build a complete model with embedding layers, MoBA attention layers, feedforward networks, and output layers.
* **Performance Optimization:**
    * **SSD Access Patterns:**  Optimize how you load and save the KV cache to minimize SSD read/write latency. Consider using memory mapping or specialized SSD-optimized formats if performance becomes critical.
    * **GPU Acceleration:** Ensure that all tensor operations, including MoBA attention and cache loading (after loading to CPU first), are efficiently moved to and performed on the GPU if you are using one.
* **More Sophisticated Gating:** Explore different gating mechanisms as mentioned in the MoBA paper or other Mixture of Experts literature.
* **Experiment with `moba_chunk_size` and `moba_topk`:**  These hyperparameters can significantly impact performance and efficiency. Experiment to find optimal values for your specific use case and dataset.

This detailed plan and the conceptual code structure should give you a solid foundation to start building your MoBA sample with a custom SSD-based KV cache. Remember that this is a starting point, and building a fully functional, high-performance system will require further development and optimization.