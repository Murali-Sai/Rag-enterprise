"""Query dashboard — a Streamlit client for the RAG API.

A separate process that talks to the API over HTTP, exactly as any other
client would. It holds no retrieval logic, no scoring, and no thresholds: if
a number appears here it came off a response body, because a dashboard that
recomputed anything would be able to disagree with the system it is meant to
show.

It traces one question through the stages that produced the answer — which
departments the role could search and which walls removed the rest, the
ranked chunks with their scores, the answer or the structured refusal, the
per-claim citation verdicts, the confidence breakdown, the guardrails — and
can run the same question through dense and hybrid retrieval side by side.

Three rules the layout keeps:

  1. A refusal is not an error. All three unanswerable strata score a perfect
     1.000 in the evaluation; rendering a refusal in red would tell the reader
     the opposite of what the measurement says.
  2. No number without its scale. The confidence composite never appears
     without its label, a null retrieval score renders as unavailable rather
     than as zero, and a chunk's relevance sits beside its raw score and the
     stage that produced it.
  3. Nothing is precomputed. There are no scores in this file.

Run:  streamlit run dashboard/app.py
Env:  RAG_API_URL (default http://localhost:8000)
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120

# Seeded by scripts/seed_users.py. Demo credentials for a demo corpus — the
# same four the API's OpenAPI description documents publicly.
ROLES = [
    ("Research Analyst", "research_analyst", "research1!", "Behind two walls"),
    ("Trading Desk", "trader_desk", "trade1234!", "No wall applies"),
    ("Compliance Officer", "compliance_officer", "compl1234!", "No wall applies"),
    ("Admin", "admin_user", "admin1234!", "Unfiltered"),
]

# Drawn from the 54-question evaluation set, one per stratum that exercises a
# distinct path. The labels are the strata the evaluation scores them under.
SAMPLES = [
    ("exact figure", "What was Apple's total net revenue for its most recent fiscal year?"),
    (
        "comparative",
        "Compare the risk factor disclosures between JPMorgan and Goldman Sachs "
        "regarding credit risk.",
    ),
    (
        "rbac blocked",
        "What pre-trade controls and order entry limits does ACME Financial Holdings' "
        "internal trading desk procedures manual set?",
    ),
    ("no answer", "How many iPhone units did Apple ship in fiscal 2025?"),
    (
        "out of corpus",
        "What was Netflix's total streaming revenue for its most recent fiscal year?",
    ),
    ("ambiguous", "What are the company's main risks?"),
]

SCORE_TYPES = {
    "cross_encoder": ("cross-encoder logit", "roughly −11 to +11, squashed through a logistic"),
    "cosine_relevance": ("cosine relevance", "already 0–1; a good match sits near 0.5, not 1.0"),
    "rrf": ("reciprocal rank fusion", "ordinal — carries no relevance"),
}

DECLINE_GLOSS = {
    "low_retrieval_confidence": (
        "The retrieval gate fired before generation. Nothing was sent to the model and no "
        "answer was written — the best passage scored under the insufficient-context "
        "threshold. This is a retrieval failure, and the cheapest one: it costs nothing."
    ),
    "model_refused": (
        "Retrieval cleared its threshold and the model still declined. The passages were on "
        "topic but did not state what the question asked for — a corpus or chunking gap "
        "rather than a retrieval one."
    ),
}

VERDICT_ICON = {
    "supported": "🟢",
    "partial": "🟡",
    "unsupported": "🔴",
    "out_of_range": "🔴",
}

CSS = """
<style>
  .stApp { background:#07090b; }
  section.main div[data-testid="stMarkdownContainer"] { color:rgba(232,236,232,.86); }
  h1, h2, h3 { color:#efe9da !important; }
  code { color:#f5a623 !important; }
  .stage-note { font-family:'IBM Plex Mono',monospace; font-size:12px;
                color:rgba(232,236,232,.45); }
  .chunk { border:1px solid rgba(255,255,255,.09); border-radius:4px;
           padding:9px 12px; margin-bottom:6px; }
  .chunk .prov { font-family:monospace; font-size:12px; color:rgba(232,236,232,.6); }
  .chunk .snip { font-size:12px; color:rgba(232,236,232,.4); margin-top:4px; }
  .barrier { border:1px solid rgba(192,57,43,.4); border-left:3px solid #c0392b;
             border-radius:4px; padding:10px 13px; margin-bottom:7px;
             background:rgba(192,57,43,.06); }
  .declined { border:1px solid rgba(122,162,247,.35); border-left:3px solid #7aa2f7;
              border-radius:4px; padding:14px 16px; background:rgba(122,162,247,.06); }
</style>
"""


# ── API ───────────────────────────────────────────────────────


def login(username: str, password: str) -> str:
    response = requests.post(
        f"{API_URL}/auth/token",
        json={"username": username, "password": password},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_access(token: str) -> dict:
    response = requests.get(
        f"{API_URL}/access",
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def ask(token: str, question: str, mode: str) -> dict:
    response = requests.post(
        f"{API_URL}/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": question, "retrieval_mode": mode},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def list_documents(token: str) -> dict:
    response = requests.get(
        f"{API_URL}/documents",
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ── rendering ─────────────────────────────────────────────────


def render_access(access: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.caption("DEPARTMENTS THIS ROLE MAY SEARCH")
        st.write(" ".join(f"`{d}`" for d in access["accessible_departments"]))
        if access["unrestricted"]:
            st.caption(
                "Admin: no where-clause is applied at all. The list above is what the "
                "roles grant, not what retrieval filtered on."
            )
    with right:
        st.caption("INFORMATION BARRIERS IN FORCE")
        if not access["information_barriers"]:
            st.write("_None — no Chinese Wall applies to this role._")
        for barrier in access["information_barriers"]:
            st.markdown(
                f"<div class='barrier'><b>{barrier['name']}</b><br>"
                f"<span style='font-size:12px'>{barrier['description']}</span><br>"
                f"<span style='font-size:11px;opacity:.6'>removes: "
                f"{', '.join(barrier['blocked_departments'])}</span></div>",
                unsafe_allow_html=True,
            )


def render_retrieval(data: dict) -> None:
    sources = data.get("sources") or []
    config = data.get("retrieval") or {}
    if not sources:
        st.markdown(
            "<div class='stage-note'>Nothing within this role's reach matched — the "
            "where-clause returned an empty set, so there was nothing to generate "
            "from.</div>",
            unsafe_allow_html=True,
        )
        return

    score_type = next((s.get("score_type") for s in sources if s.get("score_type")), None)
    name, scale = SCORE_TYPES.get(score_type, ("no score channel", ""))
    stages = f"{config.get('mode', '?')}"
    if config.get("reranked"):
        stages += " + cross-encoder rerank"
    if config.get("hyde"):
        stages += " + HyDE"
    st.markdown(
        f"<div class='stage-note'>{len(sources)} chunks · pipeline: {stages} · "
        f"ranked by {name}{f' ({scale})' if scale else ''}</div>",
        unsafe_allow_html=True,
    )

    for i, source in enumerate(sources, start=1):
        relevance = source.get("relevance_score")
        raw = source.get("raw_score")
        if relevance is None:
            score = "no relevance — ordinal stage" if score_type else "unscored"
        else:
            score = f"**{relevance:.3f}**"
        if raw is not None:
            score += f" · raw {raw:.2f}"
        provenance = " · ".join(
            str(p)
            for p in (
                source.get("ticker"),
                source.get("filing_type"),
                source.get("filing_date"),
                source.get("section_name"),
            )
            if p
        ) or source.get("source", "")
        st.markdown(
            f"<div class='chunk'><span class='prov'>[{i}] {provenance} "
            f"<span style='opacity:.5'>— {source.get('department', '')}</span></span><br>"
            f"<span class='snip'>{source.get('content', '')[:200]}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='stage-note'>&nbsp;&nbsp;{score}</div>", unsafe_allow_html=True)


def render_answer(data: dict) -> None:
    unanswered = data.get("unanswered")
    answer = data.get("answer", "")

    if not unanswered:
        st.markdown("##### Answer")
        st.markdown(answer)
        return

    st.markdown("##### Declined")
    # On the low-confidence path the answer *is* the report rendered as prose,
    # so printing both would show the same refusal twice. On model_refused the
    # answer is the model's own wording and is worth keeping.
    if not answer.startswith(unanswered["summary"]):
        st.markdown(answer)

    gloss = DECLINE_GLOSS.get(unanswered["reason"], "")
    st.markdown(
        f"<div class='declined'><b>declined · {unanswered['reason']}</b><br><br>"
        f"{unanswered['summary']}<br><br>"
        f"<span style='font-size:12px;opacity:.7'>{gloss}</span><br><br>"
        "<span style='font-size:12px;opacity:.7'>A refusal is a correct outcome here, not a "
        "failure: the evaluation scores the three unanswerable strata 1.000, and it is "
        "over-refusal on answerable questions that is the open defect.</span></div>",
        unsafe_allow_html=True,
    )
    if unanswered.get("searched"):
        st.caption("PASSAGES CONSULTED, BEST MATCH FIRST")
        for item in unanswered["searched"]:
            st.markdown(f"- `{item}`")
    if unanswered.get("suggested_documents"):
        st.caption("WORTH OPENING BY HAND")
        for item in unanswered["suggested_documents"]:
            st.markdown(f"- `{item}`")


def render_claims(data: dict) -> None:
    claims = data.get("claims") or []
    declined = bool(data.get("unanswered"))
    cited = any(c["cited_documents"] for c in claims)
    # A full refusal parses into sentences that rest on nothing, because they
    # assert nothing. A partial refusal keeps the panel: those citations are
    # the half of the question the system did answer.
    if not claims or (declined and not cited):
        return

    verified = any(c.get("verdict") for c in claims)
    st.markdown("##### Citations")
    st.markdown(
        f"<div class='stage-note'>{len(claims)} claim{'' if len(claims) == 1 else 's'} · "
        + (
            "each cited pairing judged separately"
            if verified
            else "verification is off on served queries (one LLM call per citation); "
            "the evaluation forces it on"
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    for claim in claims:
        verdict = claim.get("verdict") or "unverified"
        icon = VERDICT_ICON.get(verdict, "⚪")
        blocks = " ".join(f"`[{n}]`" for n in claim["cited_documents"]) or "_no citation_"
        broken = " ".join(f"`[{n}]` fabricated" for n in claim["invalid_citations"])
        st.markdown(f"{icon} **{verdict}** {blocks} {broken}")
        st.markdown(
            f"<div style='font-size:13px;opacity:.75;margin:-6px 0 10px 24px'>"
            f"{claim['claim']}</div>",
            unsafe_allow_html=True,
        )


def render_confidence(data: dict) -> None:
    confidence = data.get("confidence")
    if not confidence:
        return
    st.markdown("##### Confidence")
    # The composite never appears without its label. They can disagree: a full
    # refusal is labelled low whatever the composite says, because retrieval
    # carries half the weight and retrieval may well have gone fine.
    st.markdown(
        f"### `{confidence['label']}` &nbsp; {confidence['overall']:.3f} "
        "<span style='font-size:12px;opacity:.5'>composite</span>",
        unsafe_allow_html=True,
    )

    retrieval = confidence.get("retrieval")
    completeness = confidence["answer_completeness"]
    words = {0.0: "refused", 0.5: "partial", 1.0: "answered"}

    if retrieval is None:
        st.markdown("retrieval (×0.5) — **unavailable**")
        st.caption(
            "Unavailable, not zero — the stage that ranked these chunks produced no "
            "comparable score. Its weight is redistributed across the other two rather "
            "than counted against the answer."
        )
    else:
        st.progress(min(1.0, max(0.0, retrieval)), text=f"retrieval (×0.5) — {retrieval:.3f}")
    st.progress(
        min(1.0, max(0.0, confidence["citation_coverage"])),
        text=f"citation coverage (×0.3) — {confidence['citation_coverage']:.3f}",
    )
    st.progress(
        min(1.0, max(0.0, completeness)),
        text=f"answer completeness (×0.2) — {completeness:.3f} "
        f"{words.get(completeness, '')}",
    )

    if confidence["label"] == "low" and completeness == 0.0:
        st.caption(
            "Labelled low because the answer asserted nothing, regardless of the composite. "
            "The label is a claim about the answer, and there is no answer to be confident in."
        )


def render_guardrails(data: dict) -> None:
    flags = data.get("guardrail_flags") or []
    if not flags:
        return
    st.markdown("##### Guardrails")
    st.markdown(" ".join(f"`{flag}`" for flag in flags))


def render_result(data: dict) -> None:
    render_retrieval(data)
    st.divider()
    render_answer(data)
    st.divider()
    render_claims(data)
    render_confidence(data)
    render_guardrails(data)


# ── page ──────────────────────────────────────────────────────


def main() -> None:
    # The sidebar carries identity and the retrieval toggle — the two controls
    # nothing else works without — so it is pinned open rather than left to
    # Streamlit's "auto", which collapses it below a viewport width this
    # layout otherwise fits in.
    st.set_page_config(
        page_title="RAG Enterprise — Query Dashboard",
        page_icon="▚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    state = st.session_state
    state.setdefault("token", None)
    state.setdefault("access", None)
    state.setdefault("role_label", None)

    st.title("Query dashboard")
    st.markdown(
        "<div class='stage-note'>One question, traced through every stage that touched it. "
        f"API: <code>{API_URL}</code> — nothing on this page is precomputed.</div>",
        unsafe_allow_html=True,
    )

    # ── sidebar: identity ─────────────────────────────────────
    with st.sidebar:
        st.subheader("00 · Identity")
        labels = [r[0] for r in ROLES]
        choice = st.radio("Sign in as", labels, index=0, captions=[r[3] for r in ROLES])
        # Re-authenticate whenever the selection no longer matches who is
        # signed in. Gating this on the button alone left the radio looking
        # live while the session still held the previous role's token — and
        # switching roles mid-question is the demonstration this page exists
        # for, so it has to happen on the switch, not on a second click.
        stale = state.role_label != choice
        if stale or st.button("Re-authenticate", use_container_width=True):
            label, username, password, _ = next(r for r in ROLES if r[0] == choice)
            try:
                state.token = login(username, password)
                state.access = get_access(state.token)
                state.role_label = label
            except requests.RequestException as error:
                state.token = None
                state.role_label = None
                st.error(f"Login failed: {error}")

        if state.token:
            st.success(f"Authenticated as `{state.access['username']}`")

        st.divider()
        st.subheader("Retrieval")
        mode = st.radio(
            "Search stage",
            ["default", "dense", "hybrid", "compare dense vs hybrid"],
            help=(
                "dense = vector search only. hybrid = vector + BM25 fused by RRF. "
                "default follows the server's HYBRID_SEARCH_ENABLED, which ships off."
            ),
        )
        if mode == "compare dense vs hybrid":
            st.caption(
                "Runs the question twice — two generation calls. On this corpus the "
                "measured difference between the two is inside the run-to-run noise "
                "floor, which is the result, not a bug."
            )

        st.divider()
        if state.token and st.checkbox("Show indexed documents"):
            try:
                index = list_documents(state.token)
                st.caption(
                    f"{index['total_documents']} documents · "
                    f"{index['total_chunks']} chunks visible to this role"
                )
                for document in index["documents"]:
                    st.markdown(
                        f"`{document['source']}` — {document['chunks']} chunks "
                        f"({document['department']})"
                    )
            except requests.RequestException as error:
                st.error(f"Could not list documents: {error}")

    if not state.token:
        st.warning("Authenticate in the sidebar to begin.")
        return

    # ── access profile ────────────────────────────────────────
    st.subheader(f"Access profile — {state.role_label}")
    render_access(state.access)
    st.divider()

    # ── query ─────────────────────────────────────────────────
    st.subheader("01 · Query")
    st.markdown(
        "<div class='stage-note'>Six questions from the 54-question evaluation set, each "
        "chosen because it exercises a different path. The labels are the strata the "
        "evaluation scores them under.</div>",
        unsafe_allow_html=True,
    )
    labelled = [f"[{tag}]  {text}" for tag, text in SAMPLES]
    picked = st.selectbox("Pick a question", ["— type my own —", *labelled])
    default_text = "" if picked.startswith("—") else picked.split("]", 1)[1].strip()
    question = st.text_input("Question", value=default_text, placeholder="ask the filing index…")

    if not st.button("Run", type="primary"):
        return
    if not question.strip():
        st.warning("Enter a question.")
        return

    if mode == "compare dense vs hybrid":
        left, right = st.columns(2, border=True)
        for column, pinned in ((left, "dense"), (right, "hybrid")):
            with column:
                st.markdown(f"### {pinned}")
                with st.spinner(f"querying ({pinned})…"):
                    try:
                        render_result(ask(state.token, question, pinned))
                    except requests.RequestException as error:
                        st.error(f"Query failed: {error}")
        return

    with st.spinner("querying the filing index…"):
        try:
            render_result(ask(state.token, question, mode))
        except requests.RequestException as error:
            st.error(f"Query failed: {error}")


if __name__ == "__main__":
    main()
