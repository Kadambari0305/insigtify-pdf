from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["document_analysis"]
documents_collection = db["documents"]

# Filename to search for
filename = "Details.pdf"

# Search for the document (case-insensitive)
document = documents_collection.find_one({"filename": {"$regex": f"^{filename}$", "$options": "i"}})

# Output the result
if document:
    print("✅ Document found:", document)
else:
    print("❌ No document found!")
