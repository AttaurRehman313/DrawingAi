import streamlit as st
import difflib
import os
import time
from PIL import Image
from new import dict_new  # Assuming you have this module with the dictionary dict_new
import google.generativeai as genai
import os
from dotenv import load_dotenv

from streamlit_carousel import carousel

load_dotenv()

genai.configure(api_key="AIzaSyAMrWNi5RMAyR6La37qbJY79XqLf1HbZ-8")


# def query_process(user_query):
#     prompt = f"""You are an expert query analyzer designed to get intend word out of user query.\n
#     Your task is to find the relavent 
#     user query: {user_query}"""
#     model = genai.GenerativeModel(model_name="gemini-1.5-flash")
#     response = model.generate_content(prompt)
#     return response.text

# print("???????????????????",query_process("where is paris?"))

import json
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import base64
import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAh77CFeBM9y8HpnqBUyIyjwoVmmT9Ofng")
llm = genai.GenerativeModel('gemini-1.5-pro')


loader = CSVLoader(file_path="data.csv")

data = loader.load()

# print(data['location'])

embedding_dir = "Embedding"

def query_find(user_query,embedding_dir):
    # embeddings = AzureOpenAIEmbeddings(model="unescoai-dev-text-embedding-ada-002")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAh77CFeBM9y8HpnqBUyIyjwoVmmT9Ofng")
    embedding_storage = FAISS.load_local(embedding_dir, embeddings, allow_dangerous_deserialization=True)
    docs = embedding_storage.similarity_search(user_query, k=1)
    ind = docs[0].page_content

    print("Embedding result: >>>>>>>>>>>>",ind)
    start_index = ind.find("how")


    if start_index != -1:  # Ensure 'how' was found
        result = ind[start_index:]
    else:
        result = ""

    print(result)
    return result

def keyword_research(user_query, dictionary):
    user_query = user_query.lower()
    closest_match = difflib.get_close_matches(user_query, dictionary.keys(), n=1, cutoff=0.1)
    if closest_match:
        st.write(f"User Query: {user_query}")
        return dictionary[closest_match[0]]
    else:
        st.write("No match found.")
        return None


def handle_path(image_list):
    test_items = [ dict(
            title="",
            text=f"",
            img=img,
        )for img in image_list 
    ]
    return test_items


def display_directory_content(directory_path):
    if os.path.isdir(directory_path):
        # Introduce a 20-second delay
        st.write("Please wait, content is being generated...")
        
        list_images = []
        # Display images in sequence
        image_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.png')])
        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            append_image = list_images.append(image_path)
            # image = Image.open(image_path)
            # st.image(image, caption=image_file)
        time.sleep(15)
        carousel(items=handle_path(list_images))


        st.markdown(
            """
            <style>
            .carousel-control-prev-icon {
                background-image: url(data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='%23fff'%3E%3Cpath d='M11.354 1.646a.5.5 0 0 1 0 .708L5.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z'/%3E%3C/svg%3E)!important;
            }
            
            .carousel-control-next-icon {
                background-image: url(data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='%23fff'%3E%3Cpath d='M4.646 14.354a.5.5 0 0 1 0-.708L10.293 8 4.646 2.354a.5.5 0 1 1 .708-.708l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708 0z'/%3E%3C/svg%3E) !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Display text from the text file
        text_file = os.path.join(directory_path, os.path.basename(directory_path) + '.txt')
        if os.path.isfile(text_file):
            with open(text_file, 'r') as file:
                text_content = file.read()
                res_gen = llm.generate_content(f"""You are a drwaing teacher for kids of age 5 to 10.\n
                Your task is to make a usefull guide out of the provided text.
                Text:{text_content}""")
                st.write( res_gen.text)
        else:
            st.write("No text file found.")
    else:
        st.write("Invalid directory path.")

def main():
    st.title("DrawingAI")

    user_query = st.text_input("Enter your query:")
    if st.button("Search"):
        result_path = query_find(user_query,embedding_dir)
        if result_path:
            # st.write(f"Generated images with steps:")
            display_directory_content(result_path)

if __name__ == "__main__":
    main()



