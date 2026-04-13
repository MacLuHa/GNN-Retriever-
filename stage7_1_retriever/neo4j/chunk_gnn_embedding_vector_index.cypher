// Neo4j 5.11+ vector index for online retrieval seeds (stage7_1_retriever GNN_USE_NEO4J_EMBEDDINGS).
// Run AFTER the notebook has written Chunk.gnn_embedding.
// Set vector.dimensions to match EMBEDDING_DIM / model output (default 256).

CREATE VECTOR INDEX chunk_gnn_embedding IF NOT EXISTS
FOR (c:Chunk)
ON (c.gnn_embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 256,
  `vector.similarity_function`: 'cosine'
}};

// Verify: SHOW INDEXES YIELD name, type, state, options WHERE name = 'chunk_gnn_embedding';
