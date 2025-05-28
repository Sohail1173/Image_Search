from PIL import Image
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os

class Image_RAG:
    def __init__(self):
        self.model_name=SentenceTransformer("Clip-VIT-B-32")
        self.client=PersistentClient(path="chroma_db")
        self.collection=self.client.get_or_create_collection(name="image_embeddings")
        self.existing_data = []
        self.new_data = []
    def add_images_to_db(self, directory):

        if os.path.exists("chroma_db"):
            print("Fetching existing metadata from vector store")
            entities = self.collection.get(include=["metadatas"])
            if len(entities["metadatas"]) > 0:
                self.existing_data = [entry["source"] for entry in entities["metadatas"]]

    # Process new images
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(directory, filename)

                if image_path in self.existing_data:
                    print(f"URL {image_path} already exists in the vector store")
                    continue  # skip already existing

                self.new_data.append((filename, image_path))

    # Add new images in batches
        MAX_BATCH_SIZE = 100
        for i in range(0, len(self.new_data), MAX_BATCH_SIZE):
            batch = self.new_data[i:i + MAX_BATCH_SIZE]
            embeddings = []
            ids = []
            metadatas = []

            for filename, path in batch:
                try:
                    image = Image.open(path)
                    embedding = self.model_name.encode(image)  # Assuming encode returns vector
                    embeddings.append(embedding)
                    ids.append(filename)
                    metadatas.append({"filename": filename, "source": path})
                except Exception as e:
                    print(f"Error processing image {path}: {e}")

            if embeddings:
                self.collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Stored vectors for batch {i} to {i + len(batch)} successfully...")

   
        
def main():
    rag=Image_RAG()
    data=rag.add_images_to_db("image")

     
        
if __name__ == "__main__":
        main()