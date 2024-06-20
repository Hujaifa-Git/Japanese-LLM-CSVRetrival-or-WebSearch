from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM,pipeline, AutoTokenizer, BitsAndBytesConfig
import config as ctg
import os
import torch

os.environ['SERPAPI_API_KEY'] = ctg.SERP_API

app = Flask(__name__)

CHAIN = None
RETRIEVAL_CONTEXT_TEXT = ''
DB = None

def create_rag(csv_path):
    global CHAIN
    global DB
    
    csvLoader = CSVLoader(csv_path, encoding='utf-8')
    documents = csvLoader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20, separators='\n\n\n')
    docs = text_splitter.split_documents(documents)
    modelPath = ctg.embed_model
    model_kwargs = {'device':'cuda:0'}
    encode_kwargs = {'normalize_embeddings':False}
    embeddings = HuggingFaceEmbeddings(
    model_name = modelPath,  
    model_kwargs = model_kwargs,
    encode_kwargs=encode_kwargs
    )

    db = FAISS.from_documents(docs, embeddings)
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(ctg.base_model)
    model = AutoModelForCausalLM.from_pretrained(ctg.base_model,
                                                #  load_in_4bit=True,
                                                # quantization_config=bnb_config,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2",
                                                max_length = 64)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(
        pipeline = pipe,
        model_kwargs={"temperature": 0.0, "max_length": 64},
    )

    template = """[INST] <>
あなたは誠実で優秀な日本人アシスタントです。 私の質問に答えるには、次のマークダウン形式のコンテキスト情報と以前のチャット履歴を使用してください。 複数のコンテキストがあります。 質問と以前のチャット履歴に基づいてコンテキストを 1 つだけ選択して、回答してください。 答えがわからない場合は、答えをでっち上げようとせず、ただわからないと言ってください。 回答はできるだけ短く簡潔にしてください。
<>

チャット履歴: {history}

コンテキスト 1: {context1}
コンテキスト 2: {context2}
コンテキスト 3: {context3}

質問: {question}
[/INST]"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context1", "context2", "context3", "question"],template=template)
    memory = ConversationBufferMemory(memory_key='history', input_key="question")
    print('LLM::',llm)

    chain = LLMChain(
        llm=llm,   
        prompt=QA_CHAIN_PROMPT,
        memory=memory
    )
    
    CHAIN = chain
    DB = db
    
    


UPLOAD_FOLDER = 'uploads'  # Define the directory where CSV files will be saved
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # print('HUJAIFA')

    if request.method == 'POST':
        file = request.files['csv_file']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the uploaded file to the defined directory
                        
            create_rag(file_path)
            return "Success"

    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    global RETRIEVAL_CONTEXT_TEXT
    
    user_input = request.json['message']
    

    if user_input != '': 
        
        # RETRIEVAL_CONTEXT_TEXT += f'{user_input}\n'
        context = DB.similarity_search_with_score(user_input)
        
        llm_response = ''
        if context[0][1]<=0.3:
            response = CHAIN({
                "question": user_input, 
                "context1": context[0][0].page_content, 
                "context2": context[1][0].page_content, 
                "context3": context[2][0].page_content})
            llm_response += response['text'][response['text'].rfind('[/INST]')+7:] + f'(FROM DATABASE )' 
        else:
            retrieval = SerpAPIWrapper(params={
                "engine": "google",
                "gl": "jp",
                "hl": "ja",
            })
            result = retrieval.run(user_input)
            response = CHAIN({"question": user_input, 
                              "context1": result, 
                              "context2": '', 
                              "context3": ''})
            llm_response += response['text'][response['text'].rfind('[/INST]')+7:] + f'(FROM WEB)'

        return jsonify({'response': llm_response})



if __name__ == '__main__':
    app.run(debug=True)
