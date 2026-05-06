from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

from config import SEUIL_CONFIANCE, EMBEDDING_MODEL, MEDICAMENTS

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


# --- Mock retriever — à remplacer dès que Harald livre indexation.py ---
_MOCK_DATA = {
    "doliprane": [
        {
            "contenu": "Le doliprane contient du paracétamol 500 mg. "
                       "Posologie adulte : 1 comprimé toutes les 4 heures, maximum 3 g/jour.",
            "metadata": {"medicament": "doliprane", "section": "Posologie"},
            "score": 0.42,
        },
        {
            "contenu": "Contre-indications : insuffisance hépatique sévère, "
                       "allergie au paracétamol.",
            "metadata": {"medicament": "doliprane", "section": "Contre-indications"},
            "score": 0.61,
        },
    ],
    "ibuprofene": [
        {
            "contenu": "Ibuprofène 400 mg. Posologie adulte : 1 comprimé toutes les 6 heures "
                       "au cours des repas, maximum 3 comprimés par jour.",
            "metadata": {"medicament": "ibuprofene", "section": "Posologie"},
            "score": 0.38,
        },
        {
            "contenu": "Effets indésirables fréquents : troubles digestifs (nausées, douleurs "
                       "abdominales), maux de tête. En cas d'utilisation prolongée : risque "
                       "d'ulcère gastrique.",
            "metadata": {"medicament": "ibuprofene", "section": "Effets indésirables"},
            "score": 0.55,
        },
        {
            "contenu": "Contre-indications : antécédents d'ulcère gastro-duodénal, insuffisance "
                       "rénale sévère, grossesse à partir du 6e mois. Ne pas associer à l'aspirine.",
            "metadata": {"medicament": "ibuprofene", "section": "Contre-indications"},
            "score": 0.67,
        },
    ],
    "amoxicilline": [
        {
            "contenu": "Amoxicilline 500 mg. Antibiotique de la famille des pénicillines. "
                       "Posologie adulte : 1 gélule 3 fois par jour pendant 7 jours.",
            "metadata": {"medicament": "amoxicilline", "section": "Posologie"},
            "score": 0.40,
        },
        {
            "contenu": "Effets indésirables : diarrhées, nausées, éruptions cutanées. "
                       "En cas de réaction allergique (urticaire, gonflement), arrêter "
                       "immédiatement et consulter un médecin.",
            "metadata": {"medicament": "amoxicilline", "section": "Effets indésirables"},
            "score": 0.58,
        },
        {
            "contenu": "Contre-indications : allergie aux pénicillines ou aux céphalosporines. "
                       "Ne pas utiliser sans prescription médicale.",
            "metadata": {"medicament": "amoxicilline", "section": "Contre-indications"},
            "score": 0.72,
        },
    ],
}


def _rechercher_mock(question, _modele, _index, _chunks_avec_meta, k=4):
    question_lower = question.lower()
    for medicament, chunks in _MOCK_DATA.items():
        if medicament in question_lower:
            return chunks[:k]
    # Aucun médicament reconnu → score élevé pour déclencher le refus
    return [{"contenu": "", "metadata": {"medicament": "", "section": ""}, "score": 9.9}]


# Import de la vraie fonction de Harald dès qu'elle est disponible
try:
    from indexation import rechercher, charger_index
    _USE_MOCK = False
except ImportError:
    rechercher = _rechercher_mock
    _USE_MOCK = True


# ---------------------------------------------------------------------------
# Prompt système
# ---------------------------------------------------------------------------

def construire_prompt_systeme() -> str:
    return (
        "Tu es un assistant d'information sur les médicaments. "
        "Tu réponds UNIQUEMENT à partir du contexte fourni entre balises [Source], "
        "jamais de ta mémoire propre. "
        "Pour chaque information, indique le médicament source et la section concernée. "
        "Si le contexte ne contient pas la réponse, réponds exactement : "
        "'Je ne trouve pas cette information dans ma base de données.' "
        "N'invente jamais d'information médicale. "
        "Termine TOUJOURS chaque réponse par cette phrase exacte, sur sa propre ligne : "
        "'Ces informations ne remplacent pas l'avis d'un professionnel de santé. "
        "En cas de doute, consultez votre médecin ou votre pharmacien.'"
    )


# ---------------------------------------------------------------------------
# Construction du contexte et génération
# ---------------------------------------------------------------------------

def _assembler_contexte(chunks: list) -> str:
    parties = []
    for i, chunk in enumerate(chunks):
        med = chunk["metadata"]["medicament"]
        sect = chunk["metadata"]["section"]
        parties.append(f"[Source {i + 1} — {med} / {sect}]\n{chunk['contenu']}")
    return "\n\n".join(parties)


def generer_reponse(question: str, chunks_pertinents: list) -> str:
    contexte = _assembler_contexte(chunks_pertinents)
    messages = [
        {"role": "system", "content": construire_prompt_systeme()},
        {"role": "user", "content": f"Contexte :\n{contexte}\n\nQuestion : {question}"},
    ]
    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Seuil de pertinence (Bonus B)
# ---------------------------------------------------------------------------

def chunks_sont_pertinents(chunks: list) -> bool:
    """Distance L2 FAISS : plus petit = plus proche. On rejette si trop grand."""
    if not chunks:
        return False
    return chunks[0]["score"] < SEUIL_CONFIANCE


# ---------------------------------------------------------------------------
# Bonus D — comparaison de deux médicaments
# ---------------------------------------------------------------------------

def _detecter_deux_medicaments(question: str):
    """Retourne (med1, med2) si la question mentionne deux médicaments connus, sinon None."""
    trouves = [m for m in MEDICAMENTS if m.lower() in question.lower()]
    if len(trouves) >= 2:
        return trouves[0], trouves[1]
    return None


def _generer_comparaison(
    question: str, med1: str, med2: str, modele, index, chunks_avec_meta
) -> str:
    chunks1 = rechercher(med1, modele, index, chunks_avec_meta, k=3)
    chunks2 = rechercher(med2, modele, index, chunks_avec_meta, k=3)
    contexte = _assembler_contexte(chunks1 + chunks2)
    messages = [
        {"role": "system", "content": construire_prompt_systeme()},
        {
            "role": "user",
            "content": (
                f"Contexte :\n{contexte}\n\n"
                f"Question comparative : {question}\n"
                f"Fais une synthèse comparative entre {med1} et {med2} "
                "en te basant uniquement sur le contexte fourni."
            ),
        },
    ]
    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------

def main():
    print("Chargement de la base de connaissances...")

    if _USE_MOCK:
        print("[MODE MOCK] indexation.py absent — données de test utilisées.")
        index, chunks_avec_meta = None, []
    else:
        index, chunks_avec_meta = charger_index()

    modele = SentenceTransformer(EMBEDDING_MODEL)
    print("Système RAG prêt. Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Votre question : ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        if not question:
            continue

        # Bonus D : comparaison détectée
        paire = _detecter_deux_medicaments(question)
        if paire and not _USE_MOCK:
            med1, med2 = paire
            print(f"\n[Comparaison : {med1} vs {med2}]\n")
            reponse = _generer_comparaison(
                question, med1, med2, modele, index, chunks_avec_meta
            )
            print(reponse + "\n")
            continue

        chunks = rechercher(question, modele, index, chunks_avec_meta)

        if not chunks_sont_pertinents(chunks):
            print(
                "Je n'ai pas trouvé d'information pertinente dans ma base "
                "pour cette question.\n"
            )
            continue

        reponse = generer_reponse(question, chunks)
        print("\n" + reponse + "\n")


if __name__ == "__main__":
    main()
