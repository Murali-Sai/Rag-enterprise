/* Query dashboard.
 *
 * Everything rendered here is read off a live response. Two rules the markup
 * exists to keep:
 *
 *   1. A refusal is not an error. All three unanswerable strata score a
 *      perfect 1.000 in the evaluation; painting them red would tell a reader
 *      the opposite of what the measurement says.
 *   2. No number without its scale. `confidence.overall` never appears
 *      without its label, a null retrieval score renders as unavailable
 *      rather than as zero, and a chunk's relevance is shown next to the raw
 *      score and the stage that produced it.
 *
 * DOM is built node by node with textContent rather than assembled as HTML
 * strings: the answer, the claims and the chunk snippets are all model or
 * filing text, and one stray angle bracket in a 10-K should not be able to
 * write markup into the page.
 */

const ROLES = [
  { user: 'research_analyst',   pass: 'research1!',  label: 'Research Analyst' },
  { user: 'trader_desk',        pass: 'trade1234!',  label: 'Trading Desk' },
  { user: 'compliance_officer', pass: 'compl1234!',  label: 'Compliance Officer' },
  { user: 'admin_user',         pass: 'admin1234!',  label: 'Admin' },
];

// How to read a raw score, per retrieval stage. `relevance` is null for RRF
// on purpose — it fuses ranks and discards the underlying scores, so it says
// "this came first" and nothing about how relevant first was.
const SCORE_TYPES = {
  cross_encoder:     { name: 'cross-encoder logit', scale: 'roughly −11 to +11, squashed through a logistic' },
  cosine_relevance:  { name: 'cosine relevance',    scale: 'already 0–1; a good match sits near 0.5, not 1.0' },
  rrf:               { name: 'reciprocal rank fusion', scale: 'ordinal — carries no relevance' },
};

const DECLINE_GLOSS = {
  low_retrieval_confidence:
    'The retrieval gate fired before generation. Nothing was sent to the model and no ' +
    'answer was written — the best passage scored under the insufficient-context ' +
    'threshold. This is a retrieval failure, and it is the cheapest one: it costs nothing.',
  model_refused:
    'Retrieval cleared its threshold and the model still declined. The passages were on ' +
    'topic but did not state what the question asked for — a corpus or chunking gap ' +
    'rather than a retrieval one.',
};

let token = null;
let activeRole = null;
let lastQuestion = '';

const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function fixed3(value) {
  return Number(value).toFixed(3);
}

/* ── 00 identity ──────────────────────────────────────────── */

async function selectRole(button) {
  const { user, pass } = button.dataset;
  document.querySelectorAll('.role-btn').forEach((b) => b.classList.remove('active'));
  button.classList.add('active');

  const line = $('authLine');
  line.className = 'authline';
  line.textContent = `authenticating as ${user}…`;

  try {
    const res = await fetch('/auth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password: pass }),
    });
    if (!res.ok) throw new Error('login failed');
    token = (await res.json()).access_token;
    activeRole = user;
    line.className = 'authline ok';
    line.textContent = `authenticated as ${user}`;
    $('questionInput').disabled = false;
    $('runBtn').disabled = false;
    $('barrierHint').hidden = false;
    await loadAccess();
  } catch (e) {
    token = null;
    activeRole = null;
    line.className = 'authline err';
    line.textContent = 'authentication failed — is the server seeded? (scripts/seed_users.py)';
    $('questionInput').disabled = true;
    $('runBtn').disabled = true;
    $('accessPanel').hidden = true;
  }
}

async function loadAccess() {
  const res = await fetch('/access', { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) return;
  const access = await res.json();

  const depts = $('deptChips');
  clear(depts);
  access.accessible_departments.forEach((d) => depts.appendChild(el('span', 'chip dept', d)));
  $('unrestrictedNote').hidden = !access.unrestricted;

  const list = $('barrierList');
  clear(list);
  if (!access.information_barriers.length) {
    list.appendChild(el('div', 'barrier-none', 'None — no Chinese Wall applies to this role.'));
  } else {
    access.information_barriers.forEach((b) => {
      const card = el('div', 'barrier-card');
      card.appendChild(el('div', 'bname', b.name));
      card.appendChild(el('div', 'bdesc', b.description));
      card.appendChild(el('div', 'bblocked', `removes: ${b.blocked_departments.join(', ')}`));
      list.appendChild(card);
    });
  }
  $('accessPanel').hidden = false;
}

/* ── 01 query ─────────────────────────────────────────────── */

async function run() {
  const question = $('questionInput').value.trim();
  if (!question || !token) return;
  lastQuestion = question;

  const btn = $('runBtn');
  btn.disabled = true;
  btn.textContent = '…';
  $('results').hidden = false;
  $('answerText').textContent = '> querying the filing index…';
  $('declinedPanel').hidden = true;
  $('rerunRow').hidden = true;
  ['stage-claims', 'stage-confidence', 'stage-guardrails'].forEach((id) => { $(id).hidden = true; });
  clear($('sourceList'));
  $('retrievalMeta').textContent = '';

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ question }),
    });
    if (res.status === 401) {
      token = null;
      $('answerText').textContent = '> session expired — pick a role again';
      return;
    }
    const data = await res.json();
    if (!res.ok) {
      $('answerText').textContent = `> ${data.message || data.detail || 'query failed'}`;
      return;
    }
    render(data);
  } catch (e) {
    $('answerText').textContent = '> network error — is the server running?';
  } finally {
    btn.disabled = false;
    btn.textContent = 'RUN';
  }
}

function render(data) {
  renderRetrieval(data.sources || []);
  renderAnswer(data);
  renderClaims(data.claims || [], Boolean(data.unanswered));
  renderConfidence(data.confidence);
  renderFlags(data.guardrail_flags || []);
  renderRerun();
  $('stage-retrieval').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── 02 retrieval ─────────────────────────────────────────── */

function renderRetrieval(sources) {
  const meta = $('retrievalMeta');
  if (!sources.length) {
    meta.textContent = 'nothing within this role’s reach matched — the where-clause '
      + 'returned an empty set, so there was nothing to generate from';
    return;
  }
  const type = sources.find((s) => s.score_type)?.score_type;
  const known = SCORE_TYPES[type];
  meta.textContent = known
    ? `${sources.length} chunks · ranked by ${known.name} (${known.scale})`
    : `${sources.length} chunks · stage produced no score channel`;

  const list = $('sourceList');
  sources.forEach((s, i) => {
    const row = el('div', 'source');
    row.id = `src-${i + 1}`;
    row.appendChild(el('div', 'rank', `[${i + 1}]`));

    const body = el('div');
    const parts = [s.ticker, s.filing_type, s.filing_date, s.section_name].filter(Boolean);
    const prov = el('div', 'prov', parts.join(' · ') || s.source);
    const dept = el('span', 'dept', `  — ${s.department}`);
    prov.appendChild(dept);
    body.appendChild(prov);
    body.appendChild(el('div', 'snip', s.content));
    row.appendChild(body);

    const score = el('div', 'score');
    if (s.relevance_score === null || s.relevance_score === undefined) {
      score.appendChild(el('span', 'unscored', s.score_type ? 'no relevance—ordinal stage' : 'unscored'));
      if (s.raw_score !== null && s.raw_score !== undefined) {
        score.appendChild(el('span', 'raw', `raw ${Number(s.raw_score).toFixed(4)}`));
      }
    } else {
      score.appendChild(el('span', 'rel', fixed3(s.relevance_score)));
      if (s.raw_score !== null && s.raw_score !== undefined) {
        score.appendChild(el('span', 'raw', `raw ${Number(s.raw_score).toFixed(2)}`));
      }
    }
    row.appendChild(score);
    list.appendChild(row);
  });
}

/* ── 03 answer, or the declined outcome ───────────────────── */

function renderAnswer(data) {
  const title = $('answerTitle');
  const answer = $('answerText');
  const panel = $('declinedPanel');
  const declined = Boolean(data.unanswered);

  clear(title);
  title.className = declined ? 'stage-title is-declined' : 'stage-title';
  title.appendChild(document.createTextNode(declined ? 'Declined' : 'Answer'));
  title.appendChild(el('span', 'stage-sub',
    declined ? ' — and what it looked at' : ' — what the model wrote'));

  clear(answer);
  // On the low-confidence path `answer` *is* the report, rendered as prose by
  // render_unanswered() for the surfaces that only return a string. Showing
  // both would print the same refusal twice. On the model_refused path the
  // answer is the model's own wording and the report is added alongside it,
  // so both are worth having.
  const echoesReport = declined && (data.answer || '').startsWith(data.unanswered.summary);
  answer.hidden = echoesReport;
  if (!echoesReport) {
    answer.appendChild(citedText(data.answer || '', (data.sources || []).length));
  }

  if (!declined) {
    clear(panel);
    panel.hidden = true;
    return;
  }
  renderDeclined(data.unanswered);
}

function renderDeclined(report) {
  const panel = $('declinedPanel');
  clear(panel);
  panel.appendChild(el('div', 'dreason', `declined · ${report.reason}`));
  panel.appendChild(el('div', 'dsummary', report.summary));
  const gloss = DECLINE_GLOSS[report.reason];
  if (gloss) panel.appendChild(el('div', 'dgloss', gloss));
  panel.appendChild(el('div', 'dgloss',
    'A refusal is a correct outcome here, not a failure: the evaluation scores the ' +
    'three unanswerable strata 1.000, and it is over-refusal on answerable questions ' +
    'that is the open defect.'));

  if (report.searched && report.searched.length) {
    panel.appendChild(listBlock('Passages consulted, best match first', report.searched));
  }
  if (report.suggested_documents && report.suggested_documents.length) {
    panel.appendChild(listBlock('Worth opening by hand', report.suggested_documents));
  }
  panel.hidden = false;
}

function listBlock(label, items) {
  const block = el('div', 'dlist');
  block.appendChild(el('div', 'micro-label', label));
  const ul = el('ul');
  items.forEach((item) => ul.appendChild(el('li', null, item)));
  block.appendChild(ul);
  return block;
}

/** The answer with its bracketed citations turned into controls.
 *  A number past the end of the supplied context is a fabricated reference,
 *  and is marked as one rather than silently linked to nothing. */
function citedText(text, sourceCount) {
  const frag = document.createDocumentFragment();
  const pattern = /\[(\d+)\]/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      frag.appendChild(emphasised(text.slice(cursor, match.index)));
    }
    const n = Number(match[1]);
    const valid = n >= 1 && n <= sourceCount;
    const chip = el('button', valid ? 'cite' : 'cite broken', `[${n}]`);
    chip.type = 'button';
    if (valid) {
      chip.title = `Highlight retrieved chunk ${n}`;
      chip.addEventListener('click', () => highlightSource(n));
    } else {
      chip.title = `Block ${n} was never supplied — a fabricated reference`;
      chip.disabled = true;
    }
    frag.appendChild(chip);
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) frag.appendChild(emphasised(text.slice(cursor)));
  return frag;
}

/** `**bold**` → <strong>, and nothing else.
 *
 *  The model writes Markdown because the prompt asks for readable prose, and
 *  the asterisks are noise on screen. Deliberately not a Markdown renderer:
 *  this is LLM output over filing text, the delimiters are matched against a
 *  literal pattern, and the content between them becomes a text node. There
 *  is no path here by which the answer writes markup into the page.
 */
function emphasised(text) {
  const frag = document.createDocumentFragment();
  const pattern = /\*\*(.+?)\*\*/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      frag.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    }
    frag.appendChild(el('strong', null, match[1]));
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) frag.appendChild(document.createTextNode(text.slice(cursor)));
  return frag;
}

function highlightSource(n) {
  document.querySelectorAll('.source.lit').forEach((s) => s.classList.remove('lit'));
  const row = $(`src-${n}`);
  if (!row) return;
  row.classList.add('lit');
  row.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* ── 04 citations ─────────────────────────────────────────── */

function renderClaims(claims, declined) {
  const stage = $('stage-claims');
  // A full refusal still parses into sentences, and listing them under "what
  // each claim rests on" invites reading the refusal as an uncited assertion.
  // It rests on nothing because it asserts nothing. A *partial* refusal —
  // refusal wording plus cited claims — keeps the panel: those citations are
  // the half of the question the system did answer.
  const cited = claims.some((c) => c.cited_documents.length);
  if (!claims.length || (declined && !cited)) {
    stage.hidden = true;
    return;
  }
  const verified = claims.some((c) => c.verdict);
  const count = `${claims.length} claim${claims.length === 1 ? '' : 's'}`;
  $('claimsMeta').textContent = verified
    ? `${count} · each cited pairing judged separately`
    : `${count} · verification is off on served queries (one LLM call per citation); ` +
      'the evaluation forces it on';

  const list = $('claimList');
  clear(list);
  claims.forEach((c) => {
    const row = el('div', 'claim');
    row.appendChild(el('div', 'ctext', c.claim));

    const meta = el('div', 'cmeta');
    const verdict = c.verdict || 'unverified';
    meta.appendChild(el('span', `verdict ${verdict}`, verdict.replace(/_/g, ' ')));
    if (c.cited_documents.length) {
      c.cited_documents.forEach((n) => {
        const chip = el('button', 'cite', `[${n}]`);
        chip.type = 'button';
        chip.addEventListener('click', () => highlightSource(n));
        meta.appendChild(chip);
      });
    } else {
      meta.appendChild(el('span', 'cuncited', 'no citation'));
    }
    c.invalid_citations.forEach((n) => {
      meta.appendChild(el('span', 'cite broken', `[${n}] fabricated`));
    });
    row.appendChild(meta);
    if (c.reason) row.appendChild(el('div', 'creason', c.reason));
    list.appendChild(row);
  });
  stage.hidden = false;
}

/* ── 05 confidence ────────────────────────────────────────── */

const COMPLETENESS_WORDS = { 0: 'refused', 0.5: 'partial', 1: 'answered' };

function renderConfidence(confidence) {
  const stage = $('stage-confidence');
  if (!confidence) {
    stage.hidden = true;
    return;
  }
  const panel = $('confidencePanel');
  clear(panel);

  // The composite never appears without its label. They can disagree: a full
  // refusal is labelled low whatever the numeric composite says, because
  // retrieval carries half the weight and retrieval may well have gone fine.
  const head = el('div', 'conf-head');
  head.appendChild(el('span', `conf-label ${confidence.label}`, confidence.label));
  const overall = el('span', 'conf-overall', fixed3(confidence.overall));
  overall.appendChild(el('span', null, 'composite'));
  head.appendChild(overall);
  panel.appendChild(head);

  const parts = el('div', 'conf-parts');
  parts.appendChild(bar('retrieval (×0.5)', confidence.retrieval,
    'unavailable'));
  parts.appendChild(bar('citation coverage (×0.3)', confidence.citation_coverage));
  parts.appendChild(bar('answer completeness (×0.2)', confidence.answer_completeness,
    null, COMPLETENESS_WORDS[confidence.answer_completeness]));
  panel.appendChild(parts);

  if (confidence.retrieval === null || confidence.retrieval === undefined) {
    panel.appendChild(el('div', 'conf-note',
      'Retrieval confidence is unavailable, not zero — the stage that ranked these ' +
      'chunks produced no comparable score. Its weight is redistributed across the other ' +
      'two rather than counted against the answer.'));
  }
  if (confidence.label === 'low' && confidence.answer_completeness === 0) {
    panel.appendChild(el('div', 'conf-note',
      'Labelled low because the answer asserted nothing, regardless of the composite. ' +
      'The label is a claim about the answer, and there is no answer to be confident in.'));
  }
  stage.hidden = false;
}

function bar(name, value, unavailableText, suffix) {
  const row = el('div', 'part');
  row.appendChild(el('span', 'pname', name));
  const track = el('div', 'ptrack');
  const missing = value === null || value === undefined;
  if (!missing) {
    const fill = el('div', 'pfill');
    fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    track.appendChild(fill);
  } else {
    row.classList.add('unavailable');
  }
  row.appendChild(track);
  row.appendChild(el('span', 'pval',
    missing ? (unavailableText || 'n/a') : (suffix ? `${fixed3(value)} ${suffix}` : fixed3(value))));
  return row;
}

/* ── 06 guardrails ────────────────────────────────────────── */

function renderFlags(flags) {
  const stage = $('stage-guardrails');
  if (!flags.length) {
    stage.hidden = true;
    return;
  }
  const chips = $('flagChips');
  clear(chips);
  flags.forEach((f) => chips.appendChild(el('span', 'chip flag', f)));
  stage.hidden = false;
}

/* ── re-run as another role ───────────────────────────────── */

function renderRerun() {
  const row = $('rerunBtns');
  clear(row);
  ROLES.filter((r) => r.user !== activeRole).forEach((r) => {
    const btn = el('button', null, r.label);
    btn.type = 'button';
    btn.addEventListener('click', async () => {
      const target = document.querySelector(`.role-btn[data-user="${r.user}"]`);
      await selectRole(target);
      $('questionInput').value = lastQuestion;
      run();
    });
    row.appendChild(btn);
  });
  $('rerunRow').hidden = false;
}

/* ── wiring ───────────────────────────────────────────────── */

document.querySelectorAll('.role-btn').forEach((b) => {
  b.addEventListener('click', () => selectRole(b));
});
document.querySelectorAll('.sample').forEach((b) => {
  b.addEventListener('click', () => {
    $('questionInput').value = b.textContent.trim();
    $('questionInput').focus();
  });
});
$('runBtn').addEventListener('click', run);
$('questionInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') run();
});
