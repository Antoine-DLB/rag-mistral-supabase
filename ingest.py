import os
from dotenv import load_dotenv
from mistralai import Mistral
from supabase import create_client
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Charger les clés API depuis .env
load_dotenv()

# Initialiser les clients
mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Lire le PDF et extraire le texte de toutes les pages
reader = PdfReader("Solaire - Cours.pdf")
texte_complet = ""
for page in reader.pages:
    texte_complet += page.extract_text()

print(f"Texte extrait : {len(texte_complet)} caractères")

# Découper le texte en chunks de 500 caractères avec 50 caractères de chevauchement
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(texte_complet)

print(f"Nombre de chunks : {len(chunks)}")
print(f"Exemple de chunk : {chunks[0]}")

# Vectoriser chaque chunk et le stocker dans Supabase
for i, chunk in enumerate(chunks):
    
    # Créer l'embedding du chunk avec mistral-embed
    response = mistral.embeddings.create(
        model="mistral-embed",
        inputs=[chunk]
    )
    
    embedding = response.data[0].embedding
    
    # Stocker le chunk + son vecteur dans Supabase
    supabase.table("documents").insert({
        "content": chunk,
        "metadata": {"source": "Solaire - Cours.pdf", "chunk": i},
        "embedding": embedding
    }).execute()
    
    print(f"Chunk {i+1}/{len(chunks)} stocké")

print("Ingestion terminée !")