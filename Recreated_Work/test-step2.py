import json
import asyncio

from pathlib import Path
from itext2kg import iText2KG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize Chat Model
openai_llm_model  = ChatOpenAI(
    model="openai.gpt-6-luna-medium",
    max_tokens=1000,    
    timeout=None,
    max_retries=2,
)

# Embedding model:
openai_embeddings_model = OpenAIEmbeddings(
    model="kiconnect.qwen3-embedding-8b",
    check_embedding_ctx_length=False
)


async def build_knowledge_graph(documents):

    itext2kg = iText2KG(
        llm_model=openai_llm_model,
        embeddings_model=openai_embeddings_model
    )

    kg = await itext2kg.build_graph(
        sections=documents
    )

    return kg


async def run():    
    documents = ""
    
    with open("distilled_test_results.txt", "r", encoding="utf-8") as f:
        documents = f.readlines()
    f.close()        
        
    kg = await build_knowledge_graph(documents)
    
    entities = []
    for entity in kg.entities:
        entities.append({
            "name": entity.name,
            "label": entity.label
        })
    
    relationships = []
    for relationship in kg.relationships:
        relationships.append({
            "startEntity": {
                "name": relationship.startEntity.name,
                "label": relationship.startEntity.label
            },
            "endEntity": {
                "name": relationship.endEntity.name,
                "label": relationship.endEntity.label
            },
            "name": relationship.name
        })
    
    export_data = {
        "entities": entities,
        "relationships": relationships,
        "summary": {
            "entities": len(entities),
            "relationships": len(relationships)
        }
    }
    
    output_file = Path("kg_test_results.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            export_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Knowledge graph exported to {output_file}")
    print(f"Entities: {len(entities)}")
    print(f"Relationships: {len(relationships)}")
    f.close()
    
if __name__ == "__main__":
    asyncio.run(run())