import os
from dotenv import load_dotenv
from mistralai import Mistral
from supabase import create_client

load_dotenv()

mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def ask(question):

    # Étape 1 : vectoriser la question
    response = mistral.embeddings.create(
        model="mistral-embed",
        inputs=[question]
    )
    question_embedding = response.data[0].embedding

    # Étape 2 : chercher les chunks les plus proches dans Supabase
    resultats = supabase.rpc("match_documents", {
        "query_embedding": question_embedding,
        "match_count": 5
    }).execute()

    # Étape 3 : construire le contexte à partir des chunks récupérés
    contexte = "\n\n".join([r["content"] for r in resultats.data])

    # Étape 4 : envoyer question + contexte à Mistral
    reponse = mistral.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant qui répond uniquement en te basant sur le contexte fourni. Si la réponse n'est pas dans le contexte, dis-le clairement."
            },
            {
                "role": "user",
                "content": f"Contexte :\n{contexte}\n\nQuestion : {question}"
            }
        ]
    )

    return reponse.choices[0].message.content

question = "Quelle est la formule de la quantité annuelle d'énergie solaire produite ?"
reponse = ask(question)
print(f"Question : {question}")
print(f"Réponse : {reponse}")