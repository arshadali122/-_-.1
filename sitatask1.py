from pdfminer.high_level import extract_text 
import fitz
import os
import pickle
import time
from langchain_groq import chatgroq
from langchain.vectorstores import FAISS
from langchan.text_splitter import RecursivecharacterTextspliter
from  langchain.chains import RetrievalQA
from google.colab import files 

llm = chatgroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"

)

file_path = "faiss_store.pkl"

uploaded_files=files.upload()

def process_pdfs():
    all_text =""
    image_dir ="extracted_images"
    os.makedirs(image_dir,exist_ok=true)
     for uploaded_file in uploaded_files.keys():
       print(f"processing text from {uploaded_file}...")
       extracted_text = extract_text(uploaded_file)
       all_text +=extracted_text + "\n"
        
        print(f"Extracting images from{uploaded_file}...")
       doc=fitz.open(uploaded_file)
       for page_num in range(len(doc)):
       page=doc[page_num]
       image_list = page.get_images(full=true)

       for img_index. img in enumerate(image_list):
         xref = img[0]
         base_image = doc.extract_image(xref)
         image_bytes = base_image["image"]
         image_ext = base_image["ext"]
         image_path =os.path.join(image_dir,f"{uploaded_file}_page{page_num}_img{img_index+1}.{image_ext}")

         with open(image_path,"wb") as img_files:
           img_files.write(image_bytes)
           doc.close()


           text_splitter = RecursivecharacterTextspliter(chunk_size=1000,chunk_overlap=100)
           text_chunks=text_splitter.split_text(all_text)
           print("building embdedding  and FAISS vector  store...")
           embeddings =Hugging FaceEmbeddings(model_name="sentence-transformers/all-miniLM-L6-v2")
           vectorstore =FAISS.from_texts(text_chunks,embeddings)
           with open (file_path,"Wb") as f:
             pickle.dump(vectorstore,f)
             print ("processing  completed  Text  and images  extracted.")
             print(f"images  are saved  in :{image_dir}")
             print("FAISS index  saved to disk.")

             process_pdfs()
            
            query = input("ask question :")
           if query 
           if os.path.exists (file_path):
             print("loading FAISS index... ")
             with  open(file_path,"rb") as f:
                vectorstore= pickle.load(f)

                chain=retrievalQA.from_llm(llm=llm, Retriever=vectorstore.as_retriever() )

                print("processing your query ...")
                result =chain.run(query)
                 
                 print ("answer:")
                print (result )
