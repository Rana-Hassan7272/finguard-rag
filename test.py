from sentence_transformers import SentenceTransformer

model = SentenceTransformer("hassan7272/urdu-finance-embeddings")

emb = model.encode(["zakat ka hisab kaise karein"])
print(emb.shape)