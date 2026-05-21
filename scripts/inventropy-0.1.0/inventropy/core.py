"""
Core functionality for computing inverse entropy
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification
import openai

# ==================== Global Variables ====================
# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize embedding models (lazy loading)
_embedding_models = {}


def _get_embedding_model(embedding_method):
    """Lazy load and cache embedding models"""
    if embedding_method not in _embedding_models:
        if embedding_method == "sbert-small":
            _embedding_models[embedding_method] = SentenceTransformer(
                'paraphrase-MiniLM-L6-v2', device=device
            )
        elif embedding_method == "sbert-large":
            _embedding_models[embedding_method] = SentenceTransformer(
                'all-mpnet-base-v2', device=device
            )
        elif embedding_method == "deberta":
            # Use DebertaV2Tokenizer directly to avoid loading issues
            tokenizer = DebertaV2Tokenizer.from_pretrained(
                "microsoft/deberta-v2-xlarge-mnli"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v2-xlarge-mnli"
            )
            model.to(device).eval()
            _embedding_models[embedding_method] = (tokenizer, model)
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    return _embedding_models[embedding_method]


# ==================== Embedding Method 1: SBERT Small ====================
def sbert_embeddings_small(texts):
    """Compute embeddings using paraphrase-MiniLM-L6-v2"""
    model = _get_embedding_model("sbert-small")
    unique_texts = list(dict.fromkeys(texts))
    embeddings = {text: model.encode(text, convert_to_tensor=True) for text in unique_texts}
    return embeddings


def compute_similarity_matrix_sbert(embeddings, texts):
    """Compute similarity matrix using cosine similarity (SBERT)"""
    n = len(texts)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = abs(util.cos_sim(embeddings[texts[i]], embeddings[texts[j]]).item())
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix


# ==================== Embedding Method 2: SBERT Large ====================
def sbert_embeddings_large(texts):
    """Compute embeddings using all-mpnet-base-v2"""
    model = _get_embedding_model("sbert-large")
    unique_texts = list(dict.fromkeys(texts))
    embeddings = {text: model.encode(text, convert_to_tensor=True) for text in unique_texts}
    return embeddings


# ==================== Embedding Method 3: DeBERTa ====================
@torch.inference_mode()
def _entailment_prob(s1, s2, tokenizer, model):
    """Compute P(entail | s1→s2)"""
    inputs = tokenizer(
        s1, s2,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    return probs[0, 2].item()  # label_id==2 is entailment


def compute_bidirectional_similarity(s1, s2, tokenizer, model, method="average"):
    """
    Bidirectional entailment similarity:
      method = "average": (P12 + P21)/2
      method = "min"    : min(P12, P21)
    """
    p12 = _entailment_prob(s1, s2, tokenizer, model)
    p21 = _entailment_prob(s2, s1, tokenizer, model)
    return (p12 + p21) / 2 if method == "average" else min(p12, p21)


def compute_similarity_matrix_deberta(texts, method="average"):
    """Compute similarity matrix using DeBERTa"""
    tokenizer, model = _get_embedding_model("deberta")
    n = len(texts)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            score = compute_bidirectional_similarity(texts[i], texts[j], tokenizer, model, method)
            sim[i, j] = sim[j, i] = score
    return sim


# ==================== Unified Interface ====================
def compute_embeddings_and_similarity(texts, embedding_method="sbert-small"):
    """
    Compute embeddings and similarity matrix based on specified method
    
    Args:
        texts: List of texts
        embedding_method: "sbert-small" | "sbert-large" | "deberta"
    
    Returns:
        similarity_matrix: Similarity matrix
    """
    if embedding_method == "sbert-small":
        embeddings = sbert_embeddings_small(texts)
        return compute_similarity_matrix_sbert(embeddings, texts)
    elif embedding_method == "sbert-large":
        embeddings = sbert_embeddings_large(texts)
        return compute_similarity_matrix_sbert(embeddings, texts)
    elif embedding_method == "deberta":
        return compute_similarity_matrix_deberta(texts)
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")


# ==================== Default LLM Functions (OpenAI) ====================
def default_paraphrase_func(sentence, num_perturb=9, model="gpt-3.5-turbo", **kwargs):
    """Default paraphrase function - using OpenAI"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that paraphrases sentences."},
            {"role": "user", "content": f"Provide {num_perturb} paraphrases for this sentence: {sentence}"}
        ],
        temperature=0.7
    )
    paraphrased_text = response['choices'][0]['message']['content']
    paraphrases = [p.strip() for p in paraphrased_text.split('\n') if p.strip()]
    return paraphrases[:num_perturb]


def default_answer_func(question, num_replication=5, model="gpt-3.5-turbo", temperature=0.7, **kwargs):
    """Default answer function - using OpenAI"""
    responses = []
    for _ in range(num_replication):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"{question} Answer concisely and return only the name."}],
            temperature=temperature
        )
        responses.append(response['choices'][0]['message']['content'])
    return responses


# ==================== Step 3: Compute inv-entropy ====================
def row_normalize(matrix):
    """Row normalization"""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def compute_inv_entropy_single(questions, answers, embedding_method="sbert-small"):
    """
    Compute single inv-entropy (entropy_x_y_I)
    
    Args:
        questions: List of questions
        answers: List of answers
        embedding_method: Embedding method
    """
    # 1. Compute similarity matrices
    simX = compute_embeddings_and_similarity(questions, embedding_method)
    simY = compute_embeddings_and_similarity(answers, embedding_method)
    
    # 2. Row normalization
    px = row_normalize(simX)
    py = row_normalize(simY)
    
    # 3. Compute marginal and conditional distributions
    n = px.shape[0]
    pi_uniform = np.full((1, n), 1/n)
    
    py_marginal_I = (pi_uniform @ py).T
    px_y_I = py @ px
    px_marginal_I = (pi_uniform @ py @ py @ px).T
    
    # 4. Convert dimensions (row=x, col=y)
    px_y_I = px_y_I.T
    
    # 5. Compute entropy_x_y_I (inv-entropy)
    entropy_x_y_I = -np.sum(np.diag(px_y_I) * np.log(np.diag(px_y_I)))
    
    return entropy_x_y_I, simX, simY


def compute_inv_entropy_with_sampling(
    simX_full, 
    simY_full, 
    num_perturb=9, 
    num_replication=5, 
    num_bootstrap=10
):
    """
    Compute average inv-entropy through random sampling
    
    Args:
        simX_full: Full question similarity matrix
        simY_full: Full answer similarity matrix
        num_perturb: Number of paraphrases
        num_replication: Number of answers per question
        num_bootstrap: Number of bootstrap samples
    """
    results = []
    used_indices = set()
    np.random.seed(333)
    
    # Total questions = original + paraphrases
    total_questions = num_perturb + 1
    
    for i in range(num_bootstrap):
        # Dynamically generate keep_indices: randomly select 1 answer from each question
        while True:
            keep_indices = []
            for q in range(total_questions):
                # Answer index range for each question: [q*num_replication, (q+1)*num_replication)
                start_idx = q * num_replication
                end_idx = (q + 1) * num_replication - 1
                keep_indices.append(random.randint(start_idx, end_idx))
            
            keep_indices_tuple = tuple(keep_indices)
            
            if keep_indices_tuple not in used_indices:
                used_indices.add(keep_indices_tuple)
                break
        
        # Slice similarity matrices
        simX = simX_full[keep_indices, :][:, keep_indices]
        simY = simY_full[keep_indices, :][:, keep_indices]
        
        # Row normalization
        px = row_normalize(simX)
        py = row_normalize(simY)
        
        # Compute
        n = px.shape[0]
        pi_uniform = np.full((1, n), 1/n)
        py_marginal_I = (pi_uniform @ py).T
        px_y_I = py @ px
        px_marginal_I = (pi_uniform @ py @ py @ px).T
        
        px_y_I = px_y_I.T
        
        # Compute entropy_x_y_I
        entropy_x_y_I = -np.sum(np.diag(px_y_I) * np.log(np.diag(px_y_I)))
        results.append(entropy_x_y_I)
        
        print(f"Iteration {i+1}/{num_bootstrap}: entropy_x_y_I = {entropy_x_y_I:.4f}")
    
    mean_inv_entropy = np.mean(results)
    return mean_inv_entropy


# ==================== Main Function ====================
def calculate_inv_entropy(
    original_question,
    num_perturb=9,
    num_replication=5,
    num_bootstrap=10,
    model="gpt-3.5-turbo",
    temperature=0.7,
    embedding_method="sbert-small",
    paraphrase_func=None,
    answer_func=None
):
    """
    Complete pipeline: from original question to inv-entropy computation
    
    Args:
        original_question: Original question
        num_perturb: Number of paraphrases (default 9)
        num_replication: Number of answers per question (default 5)
        num_bootstrap: Number of bootstrap samples (default 10)
        model: Model name (default gpt-3.5-turbo)
        temperature: Sampling temperature for answering (default 0.7)
        embedding_method: Embedding method (default sbert-small)
            Options: "sbert-small", "sbert-large", "deberta"
        paraphrase_func: Custom paraphrase function (optional)
            Function signature: func(sentence, num_perturb, model, **kwargs) -> List[str]
        answer_func: Custom answer function (optional)
            Function signature: func(question, num_replication, model, temperature, **kwargs) -> List[str]
    
    Returns:
        mean_inv_entropy: Average inv-entropy from bootstrap sampling
    """
    # Use custom functions or default functions
    paraphrase_fn = paraphrase_func if paraphrase_func is not None else default_paraphrase_func
    answer_fn = answer_func if answer_func is not None else default_answer_func
    
    print("=" * 60)
    print("Step 1: Generate paraphrases...")
    print("=" * 60)
    
    # 1. Generate paraphrases
    paraphrases = paraphrase_fn(
        original_question, 
        num_perturb=num_perturb, 
        model=model
    )
    all_questions = [original_question] + paraphrases
    
    print(f"Original question: {original_question}")
    for i, p in enumerate(paraphrases, 1):
        print(f"Paraphrase {i}: {p}")
    
    print("\n" + "=" * 60)
    print(f"Step 2: Get {num_replication} answers for each question...")
    print("=" * 60)
    
    # 2. Get answers for each question
    all_questions_repeated = []
    all_answers = []
    
    for i, question in enumerate(all_questions):
        print(f"\nProcessing question {i+1}/{len(all_questions)}: {question}")
        answers = answer_fn(
            question, 
            num_replication=num_replication, 
            model=model, 
            temperature=temperature
        )
        
        all_questions_repeated.extend([question] * num_replication)
        all_answers.extend(answers)
        
        print(f"  Got {len(answers)} answers")
    
    print(f"\nTotal: {len(all_questions_repeated)} question-answer pairs")
    
    print("\n" + "=" * 60)
    print(f"Step 3: Compute similarity matrices (using {embedding_method})...")
    print("=" * 60)
    
    # 3. Compute full similarity matrices
    _, simX_full, simY_full = compute_inv_entropy_single(
        all_questions_repeated, 
        all_answers, 
        embedding_method
    )
    print(f"Question similarity matrix: {simX_full.shape}")
    print(f"Answer similarity matrix: {simY_full.shape}")
    
    print("\n" + "=" * 60)
    print(f"Step 4: Compute average inv-entropy through {num_bootstrap} bootstrap samples...")
    print("=" * 60)
    
    # 4. Compute average inv-entropy through sampling
    mean_inv_entropy = compute_inv_entropy_with_sampling(
        simX_full, 
        simY_full, 
        num_perturb=num_perturb,
        num_replication=num_replication,
        num_bootstrap=num_bootstrap
    )
    
    print("\n" + "=" * 60)
    print("Final Result")
    print("=" * 60)
    print(f"Average inv-entropy (num_bootstrap={num_bootstrap}): {mean_inv_entropy:.4f}")
    
    return mean_inv_entropy
