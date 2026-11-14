import asyncio
import pymupdf
import json

from typing import Optional
from pathlib import Path

from itext2kg import DocumentsDistiller, iText2KG
from itext2kg.itext2kg_star.models.schemas import Article

from langchain_openai import ChatOpenAI, OpenAIEmbeddings



# Initialize LLM Model
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



### The ```DocumentDistiller``` Usage
async def main():
    # Initialize the DocumentDistiller with llm model.
    document_distiller = DocumentsDistiller(llm_model = openai_llm_model)


    # List of documents to be distilled. Like:
    # documents = [
    #     "Albert Einstein developed the theory of relativity.",
    #     "Einstein was born in Ulm, Germany, in 1879.",
    #     "The theory of relativity transformed modern physics."
    #     ]
    # Or:
    documents = [
        "title - Bertology Meets Biology: Interpreting Attention in Protein Language Models",
        "authors - [['name': 'Jesse Vig', 'affiliation': 'Salesforce Research'], ['name': 'Ali Madani', 'affiliation': 'Salesforce Research'], ['name': 'Lav R. Varshney', 'affiliation': 'Salesforce Research; University of Illinois at Urbana-Champaign'], ['name': 'Caiming Xiong', 'affiliation': 'Salesforce Research'], ['name': 'Richard Socher', 'affiliation': 'Salesforce Research'], ['name': 'Nazneen Fatema Rajani', 'affiliation': 'Salesforce Research']]"
        "abstract - Transformer models learn useful representations for protein classification and generation, but their representations are difficult to interpret. This paper adapts attention-based interpretability methods to protein language models. Across BERT, ALBERT, and XLNet models trained on two protein datasets, the authors find that attention connects amino acids that are distant in sequence but close in three-dimensional structure, focuses on binding sites, and attends to progressively more complex biophysical properties in deeper layers. Attention patterns also reflect amino-acid substitution relationships. The work includes a visualization of attention in the context of protein structure."
    ]
    # Or Like this:
    # pdf_path = "./datasets/itext2kg/scientific_articles/bertology.pdf"
    # documents = [extract_pdf_text(pdf_path)]
    
    
    # Information extraction query.
    IE_query = '''
    # DIRECTIVES : 
    - Act like an experienced information extractor. 
    - You have a chunk of a scientific paper.
    - If you do not find the right information, keep its place empty.
    '''

    # Distill the documents using the defined query and output data structure.
    # Note: distill() is now async and requires await
    distilled_doc = await document_distiller.distill(documents=documents, IE_query=IE_query, output_data_structure=list)
    # distilled_doc = await document_distiller.distill(documents=documents, IE_query=IE_query, output_data_structure=Article)
    
    return distilled_doc


async def build_knowledge_graph(semantic_blocks):
    itext2kg = iText2KG(
        llm_model=openai_llm_model,
        embeddings_model=openai_embeddings_model
    )

    kg = await itext2kg.build_graph(
        sections=semantic_blocks
    )

    return kg


# Helper Function
def extract_pdf_text(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> str:
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        # Determine the page range
        start = 0 if start_page is None else start_page - 1
        end = len(pdf) if end_page is None else min(end_page, len(pdf))

        for page_number in range(start, end):
            page = pdf[page_number]
            pages.append(page.get_text("text"))

    return "\n".join(pages)


async def run():
    distilled_doc = await main()
    
    semantic_blocks = None
    if type(distilled_doc) != dict:
        semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc.model_dump().items()]
    else:
        semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc.items()]
        
    kg = await build_knowledge_graph(semantic_blocks)

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

    output_file = Path("main_results.json")
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