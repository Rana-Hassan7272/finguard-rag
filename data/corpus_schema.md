Deduplication Rule
Before indexing, cosine similarity is computed over all question embeddings.
Any pair with similarity > 0.97 is considered a near-duplicate.
The second occurrence (by source order) is removed.
Removed doc_ids are logged in corpus_stats.json.
corpus_stats.json Fields


{
  "total_source_records": 1510,
  "after_dedup": 1508,
  "removed_duplicates": 2,
  "removed_doc_ids": ["doc_0412", "doc_1023"],
  "category_counts": {},
  "difficulty_counts": {},
  "avg_question_length_words": 12.18,
  "avg_answer_length_words": 66.91,
  "build_timestamp": "ISO-8601"
}