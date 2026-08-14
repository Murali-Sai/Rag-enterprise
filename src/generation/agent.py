"""One bounded agentic retry, on the only refusal that deserves a second look.

Three things can make this system decline, and only one of them is worth
retrying:

- `ambiguous_entity` — the question needs a company and names none. Decided
  mechanically, before generation, and correct. An agent here would be slower
  and non-deterministic at a job a regex already does.
- `low_retrieval_confidence` — nothing scored above the gate. There is nothing
  to look at again.
- **`model_refused`** — retrieval looked fine and the model still declined.
  The chunks were on topic and did not contain the fact. That is the one where
  a differently-phrased search might land somewhere better, and it is the
  failure behind the last remaining refusal error in the evaluation suite
  (Goldman's principal business segments, which the model declines over good
  retrieval).

So the loop runs there and nowhere else.

**The safety property that shapes every decision below.** Of the questions
that reach `model_refused` in a full evaluation run, the large majority are
refused *correctly* — they are out-of-corpus, or the filer genuinely does not
disclose the fact. A retry that talks itself into an answer on those does not
improve the system; it converts the strongest number in the project into a
regression. This loop is therefore built to fail closed: it keeps the original
refusal unless the retry produces an answer that is *not itself a refusal*,
every search still passes through the same RBAC and entity filters (so it
cannot reach a document the caller could not already see, nor invent a company
the corpus does not hold), and whatever it produces is scored by the same
citation and confidence machinery as any other answer.

**Bounded three ways, because an unbounded agent is a billing incident.**
`agent_max_iterations` caps the number of model turns. `agent_max_tool_errors`
aborts a loop whose tools keep failing rather than letting it retry into a
wall. And a provider that cannot bind tools at all degrades to "no recovery
attempted" instead of raising — the fallback providers are free-tier models
and not all of them support tool calling.

Written as an explicit loop rather than an off-the-shelf agent executor, for
the same reason the retrieval stages are hand-written: the iteration cap, the
stop reason and the per-step record are the parts that need to be inspectable,
and a framework that hides them behind one `.run()` call would make this
stage exactly as unmeasurable as the rest of the pipeline is not.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.common.logging import get_logger
from src.config import settings
from src.generation.chains import format_documents
from src.generation.llm_factory import get_llm

logger = get_logger(__name__)

# A search callable, injected rather than imported: this module must not reach
# for a retriever of its own, or it would silently search outside the caller's
# RBAC scope. The route and the eval harness each pass the retriever they
# already built for the user.
SearchFn = Callable[[str], list[Document]]

ANSWERED = "answered"
GAVE_UP = "gave_up"
ITERATION_CAP = "iteration_cap"
TOOL_FAILURES = "tool_failures"
UNSUPPORTED = "tools_unsupported"
NO_SEARCH = "no_search_available"

_SYSTEM = """You are helping a financial analyst after a first attempt to answer \
their question from SEC 10-K filings returned nothing usable.

You have two tools:
- search_filings(query): search the indexed filings again with a rephrased query.
- list_sections(ticker): list which 10-K sections are indexed for one company.

The first attempt's passages are below. They were on topic and did not contain \
the answer, so repeating the same query will not help — rephrase toward the \
words a filing would actually use, or check which sections exist first.

Rules:
1. You may use at most {max_iterations} tool calls in total.
2. If a search returns the answer, reply with the answer, citing sources as [1], [2] \
matching the numbered passages from that search.
3. If the filings do not contain the answer, say exactly: NOTFOUND
   Declining is a correct outcome. Do not guess, and do not answer from your own \
knowledge of these companies — only from retrieved passages.

Question: {question}

Passages from the first attempt:
{context}"""


@dataclass(frozen=True)
class AgentStep:
    """One tool call and what came back — the audit trail for the loop."""

    tool: str
    arguments: dict
    observation: str
    failed: bool = False

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "observation": self.observation[:400],
            "failed": self.failed,
        }


@dataclass(frozen=True)
class AgentOutcome:
    """What the loop did, whether or not it recovered anything.

    `answer` is None whenever the original refusal should stand, which is
    every stop reason except a genuine answer. `documents` carries the
    passages the *successful* search returned, because the citations in that
    answer are numbered against them and nothing else.
    """

    stopped_because: str
    answer: str | None = None
    documents: list[Document] = field(default_factory=list)
    steps: tuple[AgentStep, ...] = ()

    @property
    def recovered(self) -> bool:
        return self.answer is not None

    def as_dict(self) -> dict:
        return {
            "stopped_because": self.stopped_because,
            "recovered": self.recovered,
            "iterations": len(self.steps),
            "steps": [s.as_dict() for s in self.steps],
        }


def _tool_specs() -> list[dict]:
    """JSON-schema tool definitions, provider-agnostic."""
    return [
        {
            "name": "search_filings",
            "description": (
                "Search the indexed SEC 10-K filings with a rephrased query. "
                "Returns numbered passages. Name the company in the query to "
                "restrict the search to its filing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The rephrased search query.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "list_sections",
            "description": (
                "List which 10-K sections are indexed for one company, with the "
                "number of passages in each. Use to decide where an answer would live."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. GS, AAPL, JPM, MSFT, TSLA.",
                    }
                },
                "required": ["ticker"],
            },
        },
    ]


def list_sections(ticker: str) -> str:
    """Sections indexed for one company, with passage counts.

    Metadata only — no embedding, no LLM, no similarity search. Reads the
    same store the retriever reads, so it cannot describe a corpus the
    system is not actually serving.
    """
    from src.retrieval.vector_store import get_vector_store

    symbol = (ticker or "").strip().upper()
    if not symbol:
        return "No ticker given."

    metadata = get_vector_store().get_all_metadata({"ticker": {"$eq": symbol}})
    if not metadata:
        return f"{symbol} is not in the index. Indexed companies: AAPL, GS, JPM, MSFT, TSLA."

    counts: dict[str, int] = {}
    for item in metadata:
        section = str(item.get("section_name") or "unknown")
        counts[section] = counts.get(section, 0) + 1

    lines = [f"{symbol}: {len(metadata)} passages"]
    lines += [f"  {name} — {n} passages" for name, n in sorted(counts.items())]
    return "\n".join(lines)


def _run_tool(name: str, arguments: dict, search: SearchFn) -> tuple[str, list[Document], bool]:
    """Dispatch one tool call. Never raises.

    A tool that fails returns its failure *as an observation*, so the model
    can react to it — the loop is bounded by iteration and error counts, not
    by exceptions propagating into the request. A traceback here would turn a
    refusal, which is a valid answer, into a 500.
    """
    try:
        if name == "search_filings":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return "search_filings needs a non-empty query.", [], True
            found = search(query)
            if not found:
                return "That search returned no passages.", [], False
            return format_documents(found), found, False

        if name == "list_sections":
            return list_sections(str(arguments.get("ticker") or "")), [], False

        return f"Unknown tool: {name}", [], True
    except Exception as exc:  # noqa: BLE001 - a tool failure must not end the request
        logger.warning(
            "agent_tool_failed", tool=name, error=str(exc), error_type=type(exc).__name__
        )
        return f"{name} failed: {type(exc).__name__}. Try a different approach.", [], True


def _calls_from(message: AIMessage) -> list[dict]:
    """Tool calls in whatever shape the provider returned them."""
    calls = getattr(message, "tool_calls", None) or []
    normalised = []
    for call in calls:
        if isinstance(call, dict):
            normalised.append(
                {
                    "name": call.get("name", ""),
                    "args": call.get("args") or call.get("arguments") or {},
                    "id": call.get("id") or "call",
                }
            )
    return normalised


def attempt_recovery(
    question: str,
    documents: list[Document],
    search: SearchFn | None,
    max_iterations: int | None = None,
) -> AgentOutcome:
    """Give a refused question one bounded second attempt.

    Returns an outcome whose `answer` is None unless the loop produced
    something that is not itself a refusal — the caller keeps its original
    structured refusal in every other case, including every failure mode.
    """
    if search is None:
        return AgentOutcome(stopped_because=NO_SEARCH)

    cap = max_iterations if max_iterations is not None else settings.agent_max_iterations
    if cap <= 0:
        return AgentOutcome(stopped_because=ITERATION_CAP)

    try:
        llm = get_llm().bind_tools(_tool_specs())
    except Exception as exc:  # noqa: BLE001 - not every provider supports tool calling
        logger.info("agent_tools_unsupported", error=str(exc), error_type=type(exc).__name__)
        return AgentOutcome(stopped_because=UNSUPPORTED)

    prompt = _SYSTEM.format(
        max_iterations=cap,
        question=question,
        context=format_documents(documents) if documents else "(none)",
    )
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]

    steps: list[AgentStep] = []
    latest_documents: list[Document] = []
    errors = 0

    for _ in range(cap):
        try:
            reply = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - a provider error keeps the refusal
            logger.warning("agent_model_call_failed", error=str(exc), error_type=type(exc).__name__)
            return AgentOutcome(stopped_because=TOOL_FAILURES, steps=tuple(steps))

        calls = _calls_from(reply) if isinstance(reply, AIMessage) else []

        if not calls:
            text = (reply.content if isinstance(reply.content, str) else str(reply.content)).strip()
            if not text or "NOTFOUND" in text.upper():
                return AgentOutcome(stopped_because=GAVE_UP, steps=tuple(steps))
            # An answer is only worth anything with the passages it cites: the
            # numbers in it index the last search, not the original retrieval.
            return AgentOutcome(
                stopped_because=ANSWERED,
                answer=text,
                documents=latest_documents or documents,
                steps=tuple(steps),
            )

        messages.append(reply)
        for call in calls:
            observation, found, failed = _run_tool(call["name"], call["args"], search)
            steps.append(
                AgentStep(
                    tool=call["name"],
                    arguments=call["args"],
                    observation=observation,
                    failed=failed,
                )
            )
            if found:
                latest_documents = found
            if failed:
                errors += 1
            messages.append(ToolMessage(content=observation, tool_call_id=call["id"]))

        if errors >= settings.agent_max_tool_errors:
            logger.info("agent_aborted_on_tool_errors", errors=errors, steps=len(steps))
            return AgentOutcome(stopped_because=TOOL_FAILURES, steps=tuple(steps))

    logger.info("agent_hit_iteration_cap", cap=cap, steps=len(steps))
    return AgentOutcome(stopped_because=ITERATION_CAP, steps=tuple(steps))
