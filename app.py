import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag import MedicamentRAG

st.set_page_config(
    page_title="Assistant Médicaments",
    page_icon="💊",
    layout="wide",
)

# ── Load RAG (once, cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement de la base de connaissances…")
def load_rag() -> MedicamentRAG:
    return MedicamentRAG()

rag = load_rag()

# ── Sidebar — patient profile ──────────────────────────────────────────────────

with st.sidebar:
    st.title("👤 Profil patient")

    use_profile = st.toggle("Activer le profil patient", value=False)

    age = st.number_input("Âge", min_value=0, max_value=120, value=None,
                          placeholder="–", disabled=not use_profile)

    pregnancy = st.pills(
        "Grossesse / allaitement",
        ["Non", "Oui"],
        default="Non",
        disabled=not use_profile,
    )

    renal = st.pills(
        "Insuffisance rénale",
        ["Non", "Oui"],
        default="Non",
        disabled=not use_profile,
    )

    hepatic = st.pills(
        "Insuffisance hépatique",
        ["Non", "Oui"],
        default="Non",
        disabled=not use_profile,
    )

    other_meds_raw = st.text_input(
        "Autres médicaments en cours",
        placeholder="paracétamol, metformine…",
        disabled=not use_profile,
    )

    allergies = st.text_input(
        "Allergies connues",
        placeholder="pénicilline, aspirine…",
        disabled=not use_profile,
    )

    if use_profile:
        st.info("Les réponses tiendront compte de votre profil.")

profile = None
if use_profile:
    profile = {
        "age": int(age) if age else None,
        "pregnancy": pregnancy == "Oui",
        "renal_insufficiency": renal == "Oui",
        "hepatic_insufficiency": hepatic == "Oui",
        "other_medications": [m.strip() for m in other_meds_raw.split(",") if m.strip()],
        "allergies": allergies,
    }

# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("💊 Assistant Médicaments")
st.caption("Posez vos questions sur les médicaments. Activez le profil patient pour une réponse personnalisée.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New input
if prompt := st.chat_input("Votre question sur un médicament…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours…"):
            answer = (
                rag.ask_with_patient_profile(prompt, profile)
                if profile
                else rag.ask(prompt)
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
