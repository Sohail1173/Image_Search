from PIL import Image
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as img
from pathlib import Path
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
# import uu
import os
import numpy as np


class Image_RAG:
    def __init__(self):
  
        self.model_name=OpenCLIPEmbeddingFunction()
        self.client=PersistentClient(path="chroma_db")
        self.image_loader = ImageLoader()
        self.collection=self.client.get_or_create_collection(name="image_embeddings",embedding_function=self.model_name,data_loader=self.image_loader)
        self.existing_data = []
        self.new_data = []
        self.category=[]
        self.count=0
    def add_images_to_db(self, directory):

        if os.path.exists("chroma_db"):
            print("Fetching existing metadata from vector store")
            entities = self.collection.get(include=["metadatas"])
            if len(entities["metadatas"]) > 0:
                self.existing_data = [entry["source"] for entry in entities["metadatas"]]
                self.category = [entry["img_category"] for entry in entities["metadatas"]]
                self.count=len([entry["img_category"] for entry in entities["metadatas"]])

        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(directory, filename)
               

                if image_path in self.existing_data:
                    print(f"URL {image_path} already exists in the vector store")
                    continue  

                # self.new_data.append(filename, image_path)
                self.new_data.append((filename,image_path))
        MAX_BATCH_SIZE = 100
        for i in range(0, len(self.new_data), MAX_BATCH_SIZE):
            batch = self.new_data[i:i + MAX_BATCH_SIZE]
            ids = []
            metadatas = []
            image_urls=[]
            img_count=len(ids)
            for filename,path in batch:
                try:
                    ids.append(filename)
                    image_urls.append(path)
                    img_count+=1
                    metadatas.append({"filename": filename, "source": path,'img_category': 'people'})
                except Exception as e:
                    print(f"Error processing image {path}: {e}")

         
            self.collection.add(
         
                ids=ids,
                metadatas=metadatas,
                uris=image_urls
            )
            print(f"Stored vectors for batch {i} to {i + len(batch)} successfully...")



    def search_image_using_images(self, query_image_path, num_results=5):
        try:
            if not isinstance(query_image_path, np.ndarray):
                raise ValueError("Image input should be a NumPy array.")

            results = self.collection.query(query_images=query_image_path, n_results=num_results)
            
            found = False
            for result, distance in zip(results['ids'][0], results["distances"][0]):
                if distance < 0.3:
                    print(f"Match: {result}, Distance: {distance}")
                    found = True
            if not found:
                print("No similar image found with distance < 0.3")
        
        except Exception as e:
            print(f"Image search error: {e}")

    def search_image_using_text(self, query_text, image_category):
        try:
            if image_category not in self.category:
                raise ValueError(f"Invalid category::'{image_category}'. Choose from: {set(self.category)}")
            
           
            
            results = self.collection.query(
                query_texts=query_text,
                n_results=self.count,
                where={'img_category': image_category}
            )
            

            if not results['ids'][0]:
                print("No matches found for your query.")
            else:
                for result in results['ids'][0]:
                    print(f"Match: {result}")

        except Exception as e:
            print(f"Text-based search error: {e}")


def main():
    rag = Image_RAG()
    rag.add_images_to_db("images")  # Assumes this function exists in your class

    while True:
        question = input("\nEnter your question or image path (or type 'quit' to exit): ").strip()
        
        if question.lower() == 'quit':
            break
        elif question.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            if not os.path.exists(question):
                print("❌ Image file not found. Please check the path.")
                continue
            try:
                image = np.array(Image.open(question))
                rag.search_image_using_images(image)
            except Exception as e:
                print(f"❌ Failed to load image: {e}")
        else:
            category = input("Enter the image category").strip().lower()
            rag.search_image_using_text(question, category)


if __name__ == "__main__":
    main()
