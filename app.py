import uuid
from pathlib import Path

import streamlit as st

from src.config import UPLOAD_DIR, TOP_K
from src.evidence.evidence_engine import (
    deduplicate_evidence,
    format_evidence_table,
)
from src.llm.ollama_client import OllamaClient
from src.models import Evidence
from src.multimodal.audio_transcriber import AudioTranscriber
from src.multimodal.image_analyzer import analyze_image
from src.rag.chunker import chunk_pages
from src.rag.document_loader import extract_pdf_pages
from src.rag.embeddings import EmbeddingService
from src.rag.retriever import LabRetriever
from src.reporting.report_generator import generate_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LabLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD CSS
# ============================================================

def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"

    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as file:
            css = file.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "style.css was not found. "
            "Make sure it exists inside the assets folder."
        )


load_css()


# ============================================================
# CACHED SERVICES
# ============================================================

@st.cache_resource
def get_ollama():
    return OllamaClient()


@st.cache_resource
def get_retriever():
    return LabRetriever(
        EmbeddingService(get_ollama())
    )


@st.cache_resource
def get_transcriber():
    return AudioTranscriber()


# ============================================================
# SESSION STATE
# ============================================================

def init_state():

    defaults = {
        "experiment_name": "",
        "collection_name": "",
        "manual_indexed": False,
        "manual_filename": "",
        "evidence": [],
        "retrieved_chunks": [],
        "chat_messages": [],
        "transcript": "",
        "report": "",
        "last_image_analysis": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# CHECK MANUAL STATUS
# ============================================================

def manual_is_ready():

    collection_name = st.session_state.get(
        "collection_name",
        "",
    )

    if not collection_name:
        return False

    try:

        exists = get_retriever().exists(
            collection_name
        )

        if exists:
            st.session_state.manual_indexed = True
            return True

    except Exception:
        pass

    return bool(
        st.session_state.get(
            "manual_indexed",
            False,
        )
    )


# ============================================================
# SAVE UPLOAD
# ============================================================

def save_upload(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix

    file_path = (
        UPLOAD_DIR
        / f"{uuid.uuid4().hex}{suffix}"
    )

    file_path.write_bytes(
        uploaded_file.getvalue()
    )

    return file_path


# ============================================================
# INDEX LAB MANUAL
# ============================================================

def index_manual(
    pdf_file,
    experiment_name,
):

    file_path = save_upload(pdf_file)

    pages = extract_pdf_pages(
        file_path
    )

    if not pages:

        raise ValueError(
            "No readable text was found in this PDF. "
            "If this is a scanned PDF, OCR support is required."
        )

    chunks = chunk_pages(
        pages
    )

    collection_name = (
        f"{experiment_name}_{pdf_file.name}"
    )

    count = get_retriever().index_chunks(
        collection_name,
        chunks,
    )

    # --------------------------------------------------------
    # SAVE INDEXING STATE
    # --------------------------------------------------------

    st.session_state.collection_name = (
        collection_name
    )

    st.session_state.manual_indexed = True

    st.session_state.manual_filename = (
        pdf_file.name
    )

    # --------------------------------------------------------
    # CREATE SOURCE EVIDENCE
    # --------------------------------------------------------

    source_evidence = []

    for chunk in chunks:

        source_evidence.append(
            Evidence(
                claim=chunk["text"][:500],
                evidence_type="SOURCE",
                source=chunk["source"],
                detail=(
                    "Information extracted from "
                    "the uploaded laboratory manual."
                ),
                confidence=1.0,
                page=chunk["page"],
            )
        )

    st.session_state.evidence = (
        deduplicate_evidence(
            source_evidence
        )
    )

    st.session_state.retrieved_chunks = []

    st.session_state.chat_messages = []

    st.session_state.report = ""

    return count, len(pages)


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(question):

    if not manual_is_ready():

        raise ValueError(
            "Please index a laboratory manual first."
        )

    chunks = get_retriever().search(
        st.session_state.collection_name,
        question,
        TOP_K,
    )

    st.session_state.retrieved_chunks = chunks

    if not chunks:

        return (
            "I could not find relevant information "
            "in the uploaded laboratory manual."
        )

    context = "\n\n".join(
        (
            f"[Page {chunk.page} — "
            f"{chunk.source}]\n"
            f"{chunk.text}"
        )
        for chunk in chunks
    )

    system_prompt = """
You are LabLens AI, an evidence-aware laboratory assistant.

Answer only using the supplied laboratory-manual context.

Rules:

1. Never invent experimental values.
2. Never invent procedures.
3. Never invent apparatus.
4. Never invent observations.
5. If information is unavailable, clearly say so.
6. Cite relevant page numbers using [Page X].
7. Keep answers clear and useful for students.
"""

    prompt = f"""
Experiment:
{st.session_state.experiment_name}

Question:
{question}

Laboratory Manual Context:

{context}
"""

    return get_ollama().chat(
        prompt,
        system=system_prompt,
        temperature=0.1,
    )


# ============================================================
# HEADER
# ============================================================

def render_header():
    st.html("""
    <div class="lablens-header">
        <div class="lablens-title">
            🔬 LabLens AI
        </div>

        <div class="lablens-subtitle">
            Multimodal AI Laboratory Assistant —
            understand experiments, analyze evidence,
            and generate structured laboratory reports.
        </div>
    </div>
    """)

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div style="
                font-size: 1.35rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            ">
                🔬 LabLens AI
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Evidence-aware multimodal laboratory assistant"
        )

        # ----------------------------------------------------
        # Ollama status
        # ----------------------------------------------------

        try:

            connected, message = (
                get_ollama().check_connection()
            )

            if connected:

                st.success(
                    "● Ollama connected"
                )

            else:

                st.error(
                    message
                )

        except Exception as exc:

            st.error(
                f"Ollama error: {exc}"
            )

        st.divider()

        # ----------------------------------------------------
        # Experiment
        # ----------------------------------------------------

        st.write(
            "**Experiment:**",
            (
                st.session_state.experiment_name
                or "Not selected"
            ),
        )

        # ----------------------------------------------------
        # Manual status
        # ----------------------------------------------------

        if manual_is_ready():

            st.success(
                "✓ Lab manual indexed"
            )

            if st.session_state.manual_filename:

                st.caption(
                    st.session_state.manual_filename
                )

        else:

            st.warning(
                "○ Lab manual not indexed"
            )

        st.divider()

        # ----------------------------------------------------
        # Evidence statistics
        # ----------------------------------------------------

        st.subheader(
            "Evidence"
        )

        counts = {}

        for item in st.session_state.evidence:

            kind = item.evidence_type

            counts[kind] = (
                counts.get(kind, 0) + 1
            )

        cols = st.columns(2)

        for index, kind in enumerate(
            [
                "SOURCE",
                "OBSERVED",
                "CALCULATED",
                "INFERRED",
                "MISSING",
            ]
        ):

            with cols[index % 2]:

                st.metric(
                    kind,
                    counts.get(kind, 0),
                )


# ============================================================
# EXPERIMENT INPUTS
# ============================================================

def render_inputs():

    st.html("""
<div class="lablens-card">
    <div class="card-title">
        📄 Laboratory Manual
    </div>

    <div class="card-description">
        Upload the experiment PDF to create the
        knowledge base used by LabLens.
    </div>
</div>
""")

    # ========================================================
    # PDF + IMAGE
    # ========================================================

    left, right = st.columns(2)

    # ========================================================
    # PDF
    # ========================================================

    with left:

        st.html("""
<div class="lablens-card">
    <div class="card-title">
        📄 Laboratory Manual
    </div>

    <div class="card-description">
        Upload the experiment PDF to create the
        knowledge base used by LabLens.
    </div>
</div>
""")

        experiment_name = st.text_input(
            "Experiment name",
            value=st.session_state.experiment_name,
            placeholder="Example: Ohm's Law",
        )

        st.session_state.experiment_name = (
            experiment_name
        )

        pdf_file = st.file_uploader(
            "Upload laboratory manual",
            type=["pdf"],
            key="lab_manual",
        )

        if st.button(
            "📚 Index Lab Manual",
            type="primary",
            disabled=(
                not experiment_name
                or pdf_file is None
            ),
            use_container_width=True,
        ):

            with st.spinner(
                "Extracting and indexing laboratory manual..."
            ):

                try:

                    count, pages = index_manual(
                        pdf_file,
                        experiment_name,
                    )

                    st.success(
                        f"✓ Successfully indexed "
                        f"{count} chunks from "
                        f"{pages} pages."
                    )

                    st.info(
                        "The laboratory manual is now ready "
                        "for questions and report generation."
                    )

                except Exception as exc:

                    st.error(
                        f"PDF indexing failed: {exc}"
                    )


    # ========================================================
    # IMAGE
    # ========================================================

    with right:

        st.html("""
<div class="lablens-card">
    <div class="card-title">
        🖼 Experiment Evidence Image
    </div>

    <div class="card-description">
        Upload a circuit, graph, result,
        calculation, or experiment photograph.
    </div>
</div>
""")

        image_file = st.file_uploader(
            "Upload experiment image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            key="experiment_image",
        )

        if st.button(
            "🔍 Analyze Image",
            disabled=image_file is None,
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing experiment image..."
            ):

                try:

                    items = analyze_image(
                        get_ollama(),
                        image_file.getvalue(),
                        image_file.name,
                    )

                    st.session_state.evidence.extend(
                        items
                    )

                    st.session_state.evidence = (
                        deduplicate_evidence(
                            st.session_state.evidence
                        )
                    )

                    st.session_state.last_image_analysis = (
                        "\n".join(
                            (
                                f"- "
                                f"[{item.evidence_type}] "
                                f"{item.claim}: "
                                f"{item.detail}"
                            )
                            for item in items
                        )
                    )

                    st.success(
                        "✓ Image analysis completed."
                    )

                except Exception as exc:

                    st.error(
                        f"Image analysis failed: {exc}"
                    )


    # ========================================================
    # AUDIO
    # ========================================================

    st.html("""
<div class="lablens-card">
    <div class="card-title">
        🎙 Observation Voice Note
    </div>

    <div class="card-description">
        Upload a voice recording describing your
        experimental observations.
    </div>
</div>
""")

    audio_file = st.file_uploader(
        "Upload observation voice note",
        type=[
            "mp3",
            "wav",
            "m4a",
            "webm",
            "ogg",
        ],
        key="observation_audio",
    )

    if st.button(
        "🎙 Transcribe Voice Note",
        disabled=audio_file is None,
        use_container_width=True,
    ):

        with st.spinner(
            "Transcribing observation..."
        ):

            try:

                audio_path = save_upload(
                    audio_file
                )

                transcript = (
                    get_transcriber().transcribe(
                        audio_path
                    )
                )

                st.session_state.transcript = (
                    transcript
                )

                if transcript:

                    st.session_state.evidence.append(
                        Evidence(
                            claim=transcript,
                            evidence_type="OBSERVED",
                            source=(
                                f"Voice note: "
                                f"{audio_file.name}"
                            ),
                            detail=(
                                "Direct transcription of "
                                "student observation."
                            ),
                            confidence=0.90,
                        )
                    )

                    st.session_state.evidence = (
                        deduplicate_evidence(
                            st.session_state.evidence
                        )
                    )

                    st.success(
                        "✓ Voice observation added."
                    )

                else:

                    st.warning(
                        "No speech was detected."
                    )

            except Exception as exc:

                st.error(
                    f"Voice transcription failed: {exc}"
                )


# ============================================================
# EVIDENCE LAYER
# ============================================================

def render_evidence():

    st.markdown(
        """
        <div class="section-title">
            2. Evidence Layer
        </div>

        <div class="section-description">
            LabLens separates information into source,
            observed, calculated, inferred, and missing evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.evidence:

        st.info(
            "No evidence collected yet. "
            "Upload a laboratory manual, image, "
            "or observation voice note."
        )

        return

    table = format_evidence_table(
        st.session_state.evidence
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    if st.session_state.transcript:

        with st.expander(
            "🎙 Full Voice Transcription"
        ):

            st.write(
                st.session_state.transcript
            )

    if st.session_state.last_image_analysis:

        with st.expander(
            "🖼 Latest Image Analysis"
        ):

            st.markdown(
                st.session_state.last_image_analysis
            )


# ============================================================
# ASK LABLENS
# ============================================================

def render_chat():

    st.markdown(
        """
        <div class="section-title">
            3. Ask LabLens
        </div>

        <div class="section-description">
            Ask questions grounded in your uploaded
            laboratory manual.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not manual_is_ready():

        st.info(
            "📄 Index a lab manual before asking "
            "grounded questions."
        )

        return

    # --------------------------------------------------------
    # Previous messages
    # --------------------------------------------------------

    for message in (
        st.session_state.chat_messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # New question
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask anything about your experiment..."
    )

    if question:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching the laboratory manual..."
            ):

                try:

                    answer = answer_question(
                        question
                    )

                    st.markdown(
                        answer
                    )

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as exc:

                    st.error(
                        f"Question answering failed: {exc}"
                    )


# ============================================================
# REPORT GENERATION
# ============================================================

def render_report():

    st.markdown(
        """
        <div class="section-title">
            4. Generate Experiment Report
        </div>

        <div class="section-description">
            Generate an evidence-aware laboratory report
            using the manual, observations, image analysis,
            and available calculations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CHECK INDEX
    # --------------------------------------------------------

    if not manual_is_ready():

        st.info(
            "📄 Index a lab manual first. "
            "The report generator will become available "
            "after successful indexing."
        )

        return

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    source_count = sum(
        1
        for item in st.session_state.evidence
        if item.evidence_type == "SOURCE"
    )

    observed_count = sum(
        1
        for item in st.session_state.evidence
        if item.evidence_type == "OBSERVED"
    )

    calculated_count = sum(
        1
        for item in st.session_state.evidence
        if item.evidence_type == "CALCULATED"
    )

    inferred_count = sum(
        1
        for item in st.session_state.evidence
        if item.evidence_type == "INFERRED"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "SOURCE",
            source_count,
        )

    with col2:
        st.metric(
            "OBSERVED",
            observed_count,
        )

    with col3:
        st.metric(
            "CALCULATED",
            calculated_count,
        )

    with col4:
        st.metric(
            "INFERRED",
            inferred_count,
        )

    st.success(
        f"✓ Manual ready: "
        f"{st.session_state.manual_filename}"
    )

    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    if st.button(
        "📑 Generate Evidence-Aware Report",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating laboratory report with Gemma..."
        ):

            try:

                chunks = (
                    st.session_state.retrieved_chunks
                )

                # --------------------------------------------
                # If user has not asked a question yet,
                # retrieve general experiment information.
                # --------------------------------------------

                if not chunks:

                    report_query = (
                        "objective theory apparatus "
                        "requirements procedure "
                        "observations calculations "
                        "result discussion conclusion "
                        f"{st.session_state.experiment_name}"
                    )

                    chunks = (
                        get_retriever().search(
                            st.session_state.collection_name,
                            report_query,
                            TOP_K,
                        )
                    )

                if not chunks:

                    st.error(
                        "No relevant content could be retrieved "
                        "from the laboratory manual."
                    )

                    return

                # --------------------------------------------
                # Generate report
                # --------------------------------------------

                report = generate_report(
                    get_ollama(),
                    st.session_state.experiment_name,
                    chunks,
                    st.session_state.evidence,
                )

                if not report:

                    st.error(
                        "The AI model returned an empty report."
                    )

                    return

                st.session_state.report = (
                    report
                )

                st.success(
                    "✓ Laboratory report generated successfully."
                )

            except Exception as exc:

                st.error(
                    f"Report generation failed: {exc}"
                )

                st.exception(exc)

    # --------------------------------------------------------
    # DISPLAY REPORT
    # --------------------------------------------------------

    if st.session_state.report:

        st.markdown("---")

        st.markdown(
            """
            <div class="section-title">
                📑 Generated Laboratory Report
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            st.session_state.report
        )

        st.download_button(
            "⬇ Download Report as Markdown",
            data=st.session_state.report,
            file_name=(
                f"{st.session_state.experiment_name}"
                "_lab_report.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.markdown(
        """
        <div class="lablens-footer">
            LabLens AI · Multimodal Evidence-Aware
            Laboratory Assistant
            <br>
            Powered by Ollama · ChromaDB · Whisper
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    init_state()

    render_sidebar()

    render_header()

    render_inputs()

    st.divider()

    render_evidence()

    st.divider()

    render_chat()

    st.divider()

    render_report()

    render_footer()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()