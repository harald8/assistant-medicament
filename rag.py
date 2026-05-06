from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

from config import SEUIL_CONFIANCE, EMBEDDING_MODEL, MEDICAMENTS, LLM_MODEL

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


# --- Mock retriever — replace once Harald delivers indexation.py ---
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


def _mock_retriever(question, _model, _index, _chunks_with_meta, k=4):
    question_lower = question.lower()
    for medication, chunks in _MOCK_DATA.items():
        if medication in question_lower:
            return chunks[:k]
    # No medication recognized → high score to trigger refusal
    return [{"contenu": "", "metadata": {"medicament": "", "section": ""}, "score": 9.9}]


# Import Harald's functions once indexation.py is available
try:
    from indexation import rechercher as retrieve, charger_index as load_index
    _USE_MOCK = False
except ImportError:
    retrieve = _mock_retriever
    _USE_MOCK = True


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
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
# Context assembly and response generation
# ---------------------------------------------------------------------------

def _build_context(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        medication = chunk["metadata"]["medicament"]
        section = chunk["metadata"]["section"]
        parts.append(f"[Source {i + 1} — {medication} / {section}]\n{chunk['contenu']}")
    return "\n\n".join(parts)


def generate_response(question: str, relevant_chunks: list) -> str:
    context = _build_context(relevant_chunks)
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"},
    ]
    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Relevance threshold (Bonus B)
# ---------------------------------------------------------------------------

def chunks_are_relevant(chunks: list) -> bool:
    """FAISS L2 distance: smaller = closer. Reject if too large."""
    if not chunks:
        return False
    return chunks[0]["score"] < SEUIL_CONFIANCE


# ---------------------------------------------------------------------------
# Bonus D — medication comparison
# ---------------------------------------------------------------------------

def _detect_two_medications(question: str):
    """Returns (med1, med2) if the question mentions two known medications, else None."""
    found = [m for m in MEDICAMENTS if m.lower() in question.lower()]
    if len(found) >= 2:
        return found[0], found[1]
    return None


def _generate_comparison(
    question: str, med1: str, med2: str, model, index, chunks_with_meta
) -> str:
    chunks1 = retrieve(med1, model, index, chunks_with_meta, k=3)
    chunks2 = retrieve(med2, model, index, chunks_with_meta, k=3)
    context = _build_context(chunks1 + chunks2)
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Contexte :\n{context}\n\n"
                f"Question comparative : {question}\n"
                f"Fais une synthèse comparative entre {med1} et {med2} "
                "en te basant uniquement sur le contexte fourni."
            ),
        },
    ]
    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    print("Chargement de la base de connaissances...")

    if _USE_MOCK:
        print("[MODE MOCK] indexation.py absent — données de test utilisées.")
        index, chunks_with_meta = None, []
    else:
        index, chunks_with_meta = load_index()

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Système RAG prêt. Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Votre question : ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        if not question:
            continue

        # Bonus D: comparison detected
        pair = _detect_two_medications(question)
        if pair and not _USE_MOCK:
            med1, med2 = pair
            print(f"\n[Comparaison : {med1} vs {med2}]\n")
            answer = _generate_comparison(question, med1, med2, model, index, chunks_with_meta)
            print(answer + "\n")
            continue

        chunks = retrieve(question, model, index, chunks_with_meta)

        if not chunks_are_relevant(chunks):
            print(
                "Je n'ai pas trouvé d'information pertinente dans ma base "
                "pour cette question.\n"
            )
            continue

        answer = generate_response(question, chunks)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    main()
