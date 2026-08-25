from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vec = embeddings.embed_query("query: ශ්‍රී ලංකාවේ ආර්ථිකය")
print(len(vec))  # should print 384