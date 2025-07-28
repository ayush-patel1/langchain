from langchain.document_loaders import TextLoader
loader=TextLoader("data.txt")
data=loader.load()

from langchain.document_loaders import CSVLoader
loader=CSVLoader("data.csv")
data=loader.load()
len(data) #this will give you the number of documents loaded

from langchain.document_loaders import UnstructuredURLLoader
loader=UnstructuredURLLoader(urls=["https://example.com","https://example2.com"])
# This will load documents from the specified URLs
data=loader.load()
len(data) #this will give you the number of documents loaded


#we have to make whole data under token limit of any llm generally near 4096
#we might think that we can use slice operator in python simply to do this
#however , it might cut certain words in the middle causing the meaning to change
#so we use text splitter to do this

from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(separator="\n",chunk_size=200, chunk_overlap=0)

chunks=splitter.split_text(data)
len(chunks) #this will give you the number of chunks created

#when we will see the size of each of the chunks it mighty vary greatly 
#this is because we have used separator as \n ,it might be that some chunks have more \n than others
#so we can use a different text splitter to make sure that each chunk is of same size


from langchain.text_splitter import RecursiveCharacterTextSplitter
#here we can specify same arguments but list of seperators
#this will make sure that each chunk is of same size
r_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=200, chunk_overlap=0)
chunks=r_splitter.split_text(data)
len(chunks) #this will give you the number of chunks created 


from sentence_transformers import SentenceTransformer
sentences=["My name is Ayush Patel and i am a software engineer",
           "I am learning about LangChain and LLMs",
           "I am also learning about vector databases and embeddings"]
encoder=SentenceTransformer('all-mpnet-base-v2')
vectors=encoder.encode(sentences)  #this will create embeddings for the sentences
vectors.shape #this will give you the shape of the vectors created
dim=vectors.shape[1]  #this will give you the dimension of the vectors created

import faiss
index=faiss.IndexFlatL2(dim)  #indexing for faster access
index.add(vectors)
search_query="I am learning about LangChain and LLMs"

vec=encoder.encode(search_query)
vec.shape  #this will give you the shape of the vector created for the search query


import numpy  as np
svec=np.array(vec).reshape(1, -1)  #reshaping the vector to match the index
D, I = index.search(svec, k=2)  #searching for the top 2 similar vectors
print("Distances:", D)  #this will give you the distances
print("Indices:", I)  #this will give you the indices of the top 2
print("Sentences:", [sentences[i] for i in I[0]])  #