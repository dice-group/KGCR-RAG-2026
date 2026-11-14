import asyncio
import pymupdf
from typing import Optional

from itext2kg import DocumentsDistiller, iText2KG
from itext2kg.itext2kg_star.models.schemas import Article

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# API_KEY and API_BASE were set through shell!

# Initialize LLM Model
openai_llm_model  = ChatOpenAI(
    model="openai.gpt-6-luna-medium",
    max_tokens=1000,    
    timeout=None,
    max_retries=2,
)

# Embedding model:
openai_embeddings_model = OpenAIEmbeddings(
    model="gwdg.qwen3-embedding-4b",
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
    # Or Like this:
    pdf_path = "./datasets/itext2kg/scientific_articles/bertology.pdf"
    documents = [extract_pdf_text(pdf_path)]
    
    
    # Information extraction query.
    IE_query = '''
    # DIRECTIVES : 
    - Act like an experienced information extractor. 
    - You have a chunk of a scientific paper.
    - If you do not find the right information, keep its place empty.
    '''

    # Distill the documents using the defined query and output data structure.
    # Note: distill() is now async and requires await
    distilled_doc = await document_distiller.distill(documents=documents, IE_query=IE_query, output_data_structure=Article)
    
    return distilled_doc


async def run():
    distilled_doc = await main()
    
    semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc.model_dump().items()]
    
    with open("distilled_test_results.txt", "w", encoding="utf-8") as f:
        f.write("Semantic Blocks")
        f.write("=" * 50 + "\n")
        
        for block in semantic_blocks:
            f.write(block + "\n")
        

################################################

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

################################################



if __name__ == "__main__":
    asyncio.run(run())