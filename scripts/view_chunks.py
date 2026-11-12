import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("documents")

data = collection.get(include=["documents", "metadatas"])

rows = sorted(
    zip(data["ids"], data["documents"], data["metadatas"]),
    key=lambda r: int(r[0].rsplit("-", 1)[-1]),
)

print("Total chunks:", len(rows))

lines = []
for id_, text, meta in rows:
    block = (
        f"\n--- {id_} | page {meta['page']} | {len(text)} chars ---\n{text}"
    )
    print(block)
    lines.append(block)

with open("chunks_dump.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved to chunks_dump.txt")