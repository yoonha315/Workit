from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)
model  = SentenceTransformer("nlpai-lab/KURE-v1")

query  = "지체상금 한도가 설정되지 않은 계약 조항"
vector = model.encode(query, normalize_embeddings=True).tolist()

response = client.query_points(
    collection_name="law_kb",
    query=vector,
    query_filter=Filter(
        must=[FieldCondition(key="is_risk_ref", match=MatchValue(value=True))]
    ),
    limit=5,
)

print(f"쿼리: {query}\n")
for p in response.points:
    chunk_id   = p.payload.get("chunk_id", "")
    chunk_text = p.payload.get("chunk_text", "")[:80]
    print(f"[{p.score:.4f}] {chunk_id} | {chunk_text}")