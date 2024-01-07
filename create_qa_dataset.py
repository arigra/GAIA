'''
create a questions and answers dataset from a large text corpus using an LLM
load the documents to memory and split into nodes using llama-index, then for each node ask the
LLM to generate questions and answers on that node 
write the formatted Q and A to a json file for training
'''
from huggingface_hub import InferenceClient
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.text_splitter import SentenceSplitter
import json

parser = SentenceSplitter(
    chunk_size=128,
    chunk_overlap=10,
    #include_extra_info=False,
    include_prev_next_rel=False,
)
documents = SimpleDirectoryReader("./data").load_data()
nodes = parser.get_nodes_from_documents(documents)


client = InferenceClient(model="http://gaia-u-01.westeurope.cloudapp.azure.com:8080")

qna_prompt = """I want you to help me create a question and an answer for training a large language model. 
    Use the provided context that contains the a paragraph of text and write a question and an answer on this context. 
    Don't write anything except the question and the answer. your reply should start with ###Human: 
    The format of your answer should be: 
    ### Human ###
    <question> 
    ### Assitant ###
    <answer> 
    """

generated_qna = []

for node in nodes:

    node_text = node.get_content()  

    query = qna_prompt + f"\n\nContext: {node_text}"
    #print(query)
    response = client.text_generation(prompt=query,max_new_tokens=1000)

    generated_text = response  
    #print(generated_text)
    split_text = generated_text.split("###Human:")  
    if len(split_text) > 1:
        question = split_text[1].split("###Assistant:")[0].strip()  
        answer = split_text[1].split("###Assistant:")[1].strip() 
    else:
        question = ""
        answer = ""
    
    print("Question: -----", question)

    print("Answer: -----", answer)

    generated_qna.append({
        "context": node_text,
        "question": question,
        "answer": answer
    })

with open("generated_qna.json", "w") as json_file:
    json.dump(generated_qna, json_file)
