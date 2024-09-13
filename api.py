import google.generativeai as genai
import os
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

genai.configure(api_key="AIzaSyAh77CFeBM9y8HpnqBUyIyjwoVmmT9Ofng")
llm = genai.GenerativeModel('gemini-1.5-flash')

embedding_dir = "Embedding"

def query_find(user_query, embedding_dir):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAh77CFeBM9y8HpnqBUyIyjwoVmmT9Ofng")
        embedding_storage = FAISS.load_local(embedding_dir, embeddings, allow_dangerous_deserialization=True)
        docs = embedding_storage.similarity_search(user_query, k=1)
        ind = docs[0].page_content

        print("Embedding result: >>>>>>>>>>>>", ind)
        start_index = ind.find("how")

        if start_index != -1:  # Ensure 'how' was found
            result = ind[start_index:]
        else:
            result = ""

        print("Result:", result)
        return result
    except Exception as e:
        print(f"Error in query_find: {e}")
        return ""

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error in encode_image: {e}")
        return ""

def display_directory_content(directory_path):
    try:
        if os.path.isdir(directory_path):
            print("Path to directory:", directory_path)
            list_images = []

            # Display images in sequence
            image_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.png')])
            for image_file in image_files:
                image_path = os.path.join(directory_path, image_file)
                encoded_image = encode_image(image_path)
                list_images.append(encoded_image)
          
            text_file = os.path.join(directory_path, os.path.basename(directory_path) + '.txt')
            if os.path.isfile(text_file):
                
                with open(text_file, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@",text_content)
                    res_gen = llm.generate_content(f"""You are a drawing teacher for kids of age 5 to 10.\n
                    Your task is to make a useful guide out of the provided text.
                    Text:{text_content}""")
                    gen_text = res_gen.text
                    #print("Generated text:", gen_text)
            else:
                gen_text = "Looks like we couldn't generate a description for this request. Feel free to try again with a different query, and we'll do our best to assist!"
                print("No text file found.")

        else:
            print("Invalid directory path.")
            return [], "It seems like we couldn't generate any images based on your request. But don't worry, we're here to help! You can try a different query or check back later for new content."

        return list_images, gen_text
    except Exception as e:
        print(f"Error in display_directory_content: {e}")
        return [], "An error occurred while processing your request."






@app.route('/', methods=['POST','GET'])
def home():
    return render_template('fourth-page.html')

@app.route('/chat', methods=['POST'])
def main():
    try:
        if request.method == 'POST':
            user_query = request.json['user_query']
            print("User query:", user_query)
            result_path = query_find(user_query, embedding_dir)
            if result_path:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>1")
                image_data= display_directory_content(f"data/{result_path}")
                print("?????????????????????????????????????2")
                return jsonify({
                    'images': image_data
                })
                
            else:
                return jsonify({'error': 'No result found.'}), 404
    except Exception as e:
        print(f"Error in main: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == "__main__":
    app.run(debug=True)

