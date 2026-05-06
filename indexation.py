import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, INDEX_PATH, CHUNKS_PATH, MEDICAMENTS

def charger_notices():
    print("Chargement du fichier Excel...")
    df = pd.read_excel("data/CIS_RCP_export.xlsx")
    
    # Garder uniquement les médicaments de notre liste
    masque = df["denomination"].str.lower().str.contains(
        "|".join(MEDICAMENTS), na=False
    )
    df_filtre = df[masque].copy()
    
    print(f"✓ {len(df_filtre)} notices trouvées pour nos médicaments.")
    return df_filtre

def decouper_en_chunks(df_filtre):
    chunks = []
    
    sections = {
        "indications": "Indications thérapeutiques",
        "posologie": "Posologie",
        "contre_indications": "Contre-indications",
        "interactions": "Interactions médicamenteuses",
        "effets_indesirables": "Effets indésirables"
    }
    
    for _, ligne in df_filtre.iterrows():
        nom = str(ligne["denomination"]).strip()
        
        for colonne, nom_section in sections.items():
            texte = str(ligne[colonne]).strip()
            
            if texte and texte != "nan" and len(texte) > 50:
                chunks.append({
                    "contenu": f"{nom} — {nom_section} : {texte}",
                    "metadata": {
                        "medicament": nom,
                        "section": nom_section
                    }
                })
    
    print(f"✓ {len(chunks)} chunks créés.")
    return chunks


def construire_index(chunks):
    print("Chargement du modèle d'embedding...")
    modele = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Génération des vecteurs...")
    textes = [chunk["contenu"] for chunk in chunks]
    vecteurs = modele.encode(textes, show_progress_bar=True)
    vecteurs = np.array(vecteurs, dtype=np.float32)
    
    print("Construction de l'index FAISS...")
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    
    print("Sauvegarde sur disque...")
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✓ {index.ntotal} vecteurs indexés.")
    return index


def charger_index():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)
    return index, chunks_avec_meta


def rechercher(question, modele, index, chunks_avec_meta, k=4):
    vecteur_q = modele.encode([question])
    vecteur_q = np.array(vecteur_q, dtype=np.float32)
    distances, indices = index.search(vecteur_q, k)
    
    resultats = []
    for dist, idx in zip(distances[0], indices[0]):
        resultats.append({
            "contenu":  chunks_avec_meta[idx]["contenu"],
            "metadata": chunks_avec_meta[idx]["metadata"],
            "score":    float(dist)
        })
    return resultats


def main():
    df_filtre = charger_notices()
    chunks = decouper_en_chunks(df_filtre)
    construire_index(chunks)
    print("\n✓ Base vectorielle prête !")


if __name__ == "__main__":
    main()