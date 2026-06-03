import json
from jiwer import wer
import statistics

# Load your JSON file
with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Store similarity scores
similarities = []

# Compute similarity for each item and update the JSON data
for item in data:
    reference = item["neapolitan"]
    hypothesis = item["transcription"]

    error = wer(reference, hypothesis)
    similarity = max(0, 1 - error)  # Similarity capped at 0 minimum

    similarity = round(similarity, 4)
    item["similarity"] = similarity
    similarities.append(similarity)

    print(f"ID: {item['id']}")
    print(f"  Neapolitan:    {reference}")
    print(f"  Transcription: {hypothesis}")
    print(f"  WER: {error:.4f}, Similarity: {similarity:.4f}")
    print()

# Summary statistics
mean_similarity = statistics.mean(similarities)
stdev_similarity = statistics.stdev(similarities) if len(similarities) > 1 else 0.0
min_similarity = min(similarities)
max_similarity = max(similarities)

print("=== Similarity Summary ===")
print(f"Mean:  {mean_similarity:.4f}")
print(f"Stdev: {stdev_similarity:.4f}")
print(f"Min:   {min_similarity:.4f}")
print(f"Max:   {max_similarity:.4f}")

# Save updated JSON (overwrite or write to new file)
with open('neapolitan_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)