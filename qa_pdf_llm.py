from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import sys

#access_token ="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"

DB_FAISS_PATH = "vectorstore/db_faiss_pdf"
loader = DirectoryLoader("csv_data/", glob="*.pdf")
data = loader.load()
print(data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'gpu'})

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)

# prepare the template we will use when prompting the AI
template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# load the language model
llm = CTransformers(model='llama-2-13b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 2048, 'temperature': 0.01, 'context_length': 2048})

# load the interpreted information from the local database
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={'device': 'cpu'})
db = FAISS.load_local("vectorstore/db_faiss_csv", embeddings)

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})


prompt = "how much projects created on '11/10/2023  3:26:15 PM'?"
output = qa_llm({'query': prompt})
print(output["result"])