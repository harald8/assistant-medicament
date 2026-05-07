from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

from config import SEUIL_CONFIANCE, EMBEDDING_MODEL, LLM_MODEL, MEDICAMENTS

load_dotenv()

# --- Mock data — replace once Harald delivers indexation.py ---
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
    for medication, chunks in _MOCK_DATA.items():
        if medication in question.lower():
            return chunks[:k]
    return [{"contenu": "", "metadata": {"medicament": "", "section": ""}, "score": 9.9}]


try:
    from indexation import rechercher as retrieve, charger_index as load_index
    _USE_MOCK = False
except ImportError:
    retrieve = _mock_retriever
    _USE_MOCK = True


class MedicamentRAG:

    def __init__(self):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        if _USE_MOCK:
            print("[MODE MOCK] indexation.py absent — données de test utilisées.")
            self._index = None
            self._chunks_with_meta = []
        else:
            self._index, self._chunks_with_meta = load_index()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        pair = self._detect_two_medications(question)
        if pair:
            med1, med2 = pair
            print(f"\n[Comparaison : {med1} vs {med2}]\n")
            return self._generate_comparison(question, med1, med2)

        effective_question = self._resolve_question(question)

        search_query = self._reformulate_query(effective_question)
        if search_query != effective_question:
            print(f"[Reformulation : {search_query}]")

        chunks = retrieve(search_query, self._model, self._index, self._chunks_with_meta, k=10)
        chunks = self._filter_by_medication(effective_question, chunks)
        chunks = self._deduplicate_by_section(chunks)

        if not self._chunks_are_relevant(chunks):
            return "Je n'ai pas trouvé d'information pertinente dans ma base pour cette question."

        return self._generate_response(question, chunks)

    def ask_with_patient_profile(self, question: str, profile: dict) -> str:
        """Answer a question taking the patient's profile into account."""
        pair = self._detect_two_medications(question)
        if pair:
            med1, med2 = pair
            print(f"\n[Comparaison : {med1} vs {med2}]\n")
            return self._generate_comparison(question, med1, med2, profile=profile)

        effective_question = self._resolve_question(question)

        search_query = self._reformulate_query(effective_question)
        if search_query != effective_question:
            print(f"[Reformulation : {search_query}]")

        chunks = retrieve(search_query, self._model, self._index, self._chunks_with_meta, k=10)
        chunks = self._filter_by_medication(effective_question, chunks)
        chunks = self._deduplicate_by_section(chunks)

        # Also retrieve prescription-condition chunks for the mentioned medication
        mentioned = [m for m in MEDICAMENTS if m.lower() in effective_question.lower()]
        if mentioned:
            rx_query = f"{mentioned[0]} conditions prescription contre-indications"
            rx_chunks = retrieve(rx_query, self._model, self._index, self._chunks_with_meta, k=5)
            rx_chunks = [
                c for c in rx_chunks
                if mentioned[0].lower() in c["metadata"]["medicament"].lower()
                and c["metadata"]["section"] in ("Conditions de prescription", "Contre-indications")
                and c not in chunks
            ]
            chunks = chunks + rx_chunks

        if not self._chunks_are_relevant(chunks):
            return "Je n'ai pas trouvé d'information pertinente dans ma base pour cette question."

        return self._generate_response_with_profile(question, chunks, profile)

    @staticmethod
    def collect_patient_profile() -> dict:
        """Interactively collect patient information via CLI questions."""
        print("\n--- Profil patient ---")
        print("Répondez aux questions suivantes (appuyez sur Entrée pour passer).\n")

        def ask(prompt: str, default: str = "") -> str:
            value = input(prompt).strip()
            return value if value else default

        age_raw = ask("Âge du patient : ")
        try:
            age = int(age_raw) if age_raw else None
        except ValueError:
            age = None

        pregnancy_raw = ask("Grossesse ou allaitement ? (oui/non) : ").lower()
        pregnancy = pregnancy_raw in ("oui", "o", "yes")

        renal_raw = ask("Insuffisance rénale ? (oui/non) : ").lower()
        renal = renal_raw in ("oui", "o", "yes")

        hepatic_raw = ask("Insuffisance hépatique ? (oui/non) : ").lower()
        hepatic = hepatic_raw in ("oui", "o", "yes")

        other_meds_raw = ask("Autres médicaments en cours (séparés par des virgules) : ")
        other_meds = [m.strip() for m in other_meds_raw.split(",") if m.strip()]

        allergies_raw = ask("Allergies connues : ")

        profile = {
            "age": age,
            "pregnancy": pregnancy,
            "renal_insufficiency": renal,
            "hepatic_insufficiency": hepatic,
            "other_medications": other_meds,
            "allergies": allergies_raw,
        }

        print()
        return profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt() -> str:
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

    @staticmethod
    def _build_context(chunks: list) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            medication = chunk["metadata"]["medicament"]
            section = chunk["metadata"]["section"]
            parts.append(f"[Source {i + 1} — {medication} / {section}]\n{chunk['contenu']}")
        return "\n\n".join(parts)

    @staticmethod
    def _chunks_are_relevant(chunks: list) -> bool:
        """FAISS L2 distance: smaller = closer. Reject if too large."""
        return bool(chunks) and chunks[0]["score"] < SEUIL_CONFIANCE

    @staticmethod
    def _detect_two_medications(question: str):
        """Returns (med1, med2) if the question mentions two known medications, else None."""
        found = [m for m in MEDICAMENTS if m.lower() in question.lower()]
        return (found[0], found[1]) if len(found) >= 2 else None

    @staticmethod
    def _filter_by_medication(question: str, chunks: list) -> list:
        """Garde uniquement les chunks du médicament mentionné dans la question."""
        mentioned = [m for m in MEDICAMENTS if m.lower() in question.lower()]
        if not mentioned:
            return chunks
        med = mentioned[0]
        filtered = [c for c in chunks if med.lower() in c["metadata"]["medicament"].lower()]
        return filtered if filtered else chunks

    @staticmethod
    def _deduplicate_by_section(chunks: list, top_n: int = 2) -> list:
        """Garde les top_n meilleurs chunks par section (score L2 le plus faible = meilleur)."""
        by_section: dict[str, list] = {}
        for chunk in chunks:
            section = chunk["metadata"]["section"]
            by_section.setdefault(section, []).append(chunk)
        result = []
        for section_chunks in by_section.values():
            sorted_chunks = sorted(section_chunks, key=lambda c: c["score"])
            result.extend(sorted_chunks[:top_n])
        return result

    def _resolve_question(self, question: str) -> str:
        """Si aucun médicament n'est cité, en identifie un depuis le symptôme et l'ajoute à la question."""
        mentioned = [m for m in MEDICAMENTS if m.lower() in question.lower()]
        if mentioned:
            return question
        med = self._identify_medication_for_symptom(question)
        if med:
            print(f"[Médicament identifié pour le symptôme : {med}]")
            return f"{question} — médicament concerné : {med}"
        return question

    def _identify_medication_for_symptom(self, question: str) -> str | None:
        """Demande au LLM quel médicament de notre liste correspond au symptôme décrit."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"Médicaments disponibles : {', '.join(MEDICAMENTS)}.\n"
                    "L'utilisateur décrit un symptôme. Réponds UNIQUEMENT avec le nom exact "
                    "d'un médicament de cette liste qui est le plus adapté à ce symptôme, "
                    "ou 'aucun' si aucun ne convient. Un seul mot, rien d'autre."
                ),
            },
            {"role": "user", "content": question},
        ]
        result = self._call_llm(messages, max_tokens=10).strip().lower()
        matched = [m for m in MEDICAMENTS if m.lower() == result]
        return matched[0] if matched else None

    def _reformulate_query(self, question: str) -> str:
        """Bonus C — reformule la question en termes médicaux pour améliorer la recherche."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui reformule des questions en mots-clés médicaux précis "
                    "pour une recherche vectorielle. Réponds uniquement avec les mots-clés, "
                    "sans phrase ni ponctuation. Exemples : "
                    "'j ai mal à la tête' → 'céphalées antidouleur posologie', "
                    "'inconvénients doliprane' → 'doliprane effets indésirables contre-indications'."
                ),
            },
            {"role": "user", "content": question},
        ]
        return self._call_llm(messages, max_tokens=30)

    def _call_llm(self, messages: list, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _generate_response(self, question: str, relevant_chunks: list) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"Contexte :\n{self._build_context(relevant_chunks)}\n\nQuestion : {question}"},
        ]
        return self._call_llm(messages, max_tokens=800)

    @staticmethod
    def _build_patient_context(profile: dict) -> str:
        lines = ["Profil du patient :"]
        if profile.get("age") is not None:
            lines.append(f"- Âge : {profile['age']} ans")
        if profile.get("pregnancy"):
            lines.append("- Grossesse ou allaitement : OUI")
        if profile.get("renal_insufficiency"):
            lines.append("- Insuffisance rénale : OUI")
        if profile.get("hepatic_insufficiency"):
            lines.append("- Insuffisance hépatique : OUI")
        if profile.get("other_medications"):
            lines.append(f"- Médicaments en cours : {', '.join(profile['other_medications'])}")
        if profile.get("allergies"):
            lines.append(f"- Allergies : {profile['allergies']}")
        return "\n".join(lines)

    def _generate_response_with_profile(self, question: str, chunks: list, profile: dict) -> str:
        patient_ctx = self._build_patient_context(profile)
        system = (
            self._build_system_prompt()
            + "\n\nEn plus du contexte médicamenteux, tu disposes du profil patient ci-dessous. "
            "Vérifie explicitement si ce profil présente des contre-indications, des précautions "
            "particulières ou des interactions avec les médicaments mentionnés. "
            "Signale tout risque spécifique au patient AVANT de répondre à la question générale."
        )
        user_content = (
            f"Contexte :\n{self._build_context(chunks)}\n\n"
            f"{patient_ctx}\n\n"
            f"Question : {question}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return self._call_llm(messages, max_tokens=1000)

    def _generate_comparison(self, question: str, med1: str, med2: str, profile: dict | None = None) -> str:
        chunks1 = retrieve(med1, self._model, self._index, self._chunks_with_meta, k=3)
        chunks2 = retrieve(med2, self._model, self._index, self._chunks_with_meta, k=3)
        context = self._build_context(chunks1 + chunks2)
        patient_section = f"\n\n{self._build_patient_context(profile)}" if profile else ""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Contexte :\n{context}{patient_section}\n\n"
                    f"Question comparative : {question}\n"
                    f"Fais une synthèse comparative entre {med1} et {med2} "
                    "en te basant uniquement sur le contexte fourni."
                ),
            },
        ]
        return self._call_llm(messages, max_tokens=1000)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    print("Chargement de la base de connaissances...")
    rag = MedicamentRAG()
    print("Système RAG prêt.\n")

    mode = input("Mode profil patient ? (oui/non) : ").strip().lower()
    profile = MedicamentRAG.collect_patient_profile() if mode in ("oui", "o", "yes") else None

    if profile:
        print("Profil enregistré. Les réponses tiendront compte de votre situation.")
    print("Tapez 'profil' pour changer de mode, 'quit' pour quitter.\n")

    while True:
        question = input("Votre question : ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        if not question:
            continue

        if question.lower() == "profil":
            profile = MedicamentRAG.collect_patient_profile()
            print("Nouveau profil enregistré.\n")
            continue

        if profile:
            print("\n" + rag.ask_with_patient_profile(question, profile) + "\n")
        else:
            print("\n" + rag.ask(question) + "\n")


if __name__ == "__main__":
    main()
