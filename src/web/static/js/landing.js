// Decorative ticker — indexed filings, not live market data.
const tickerItems = [
  'AAPL <b>10-K FY2024</b>', 'JPM <b>10-K FY2024</b>', 'TSLA <b>10-K FY2024</b>',
  'MSFT <b>10-K FY2024</b>', 'GS <b>10-K FY2024</b>', 'ITEM 1 BUSINESS',
  'ITEM 1A RISK FACTORS', 'ITEM 7 MD&A', 'ITEM 7A MARKET RISK', 'ITEM 8 FINANCIALS'
];
const track = document.getElementById('tickerTrack');
const row = tickerItems.map(t => `<span>${t}</span>`).join('');
track.innerHTML = row + row; // duplicated for seamless loop

let token = null;

async function selectRole(el, username, password) {
  document.querySelectorAll('.role-btn').forEach(b => b.classList.remove('active'));
  el.classList.add('active');

  const status = document.getElementById('statusLine');
  status.className = 'status-line';
  status.innerHTML = '&gt; authenticating as ' + username + '...<span class="cursor"></span>';

  try {
    const res = await fetch('/auth/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    if (!res.ok) throw new Error('Login failed');
    const data = await res.json();
    token = data.access_token;
    status.className = 'status-line ok';
    status.innerHTML = '&gt; authenticated as ' + username + ' — clearance: ' + el.querySelector('.raccess').textContent + '<span class="cursor"></span>';
    document.getElementById('sendBtn').disabled = false;
    document.querySelector('.term-titlebar').lastChild.textContent = ' sec-edgar-rag — role: ' + username;
  } catch (e) {
    token = null;
    status.className = 'status-line err';
    status.innerHTML = '&gt; authentication failed — try again<span class="cursor"></span>';
    document.getElementById('sendBtn').disabled = true;
  }
}

function setQuestion(el) {
  document.getElementById('questionInput').value = el.textContent;
  document.getElementById('questionInput').focus();
}

async function sendQuery() {
  const input = document.getElementById('questionInput');
  const question = input.value.trim();
  if (!question || !token) return;

  const btn = document.getElementById('sendBtn');
  const area = document.getElementById('responseArea');
  const answer = document.getElementById('responseAnswer');
  const meta = document.getElementById('responseMeta');

  btn.disabled = true;
  btn.textContent = '...';
  area.classList.add('visible');
  answer.textContent = '> querying filing index...';
  meta.innerHTML = '';

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
      body: JSON.stringify({ question })
    });

    if (res.status === 401) {
      answer.textContent = '> session expired — select a role again';
      token = null;
      return;
    }

    const data = await res.json();
    if (!res.ok) {
      answer.textContent = '> error: ' + (data.message || data.detail || 'query failed');
      return;
    }

    answer.textContent = data.answer;

    let metaHtml = '';
    if (data.guardrail_flags && data.guardrail_flags.length > 0) {
      metaHtml += '<div class="flags">';
      data.guardrail_flags.forEach(f => { metaHtml += '<span class="flag">' + f + '</span>'; });
      metaHtml += '</div>';
    }
    if (data.sources && data.sources.length > 0) {
      const id = 'src-' + Date.now();
      metaHtml += '<button class="src-toggle" onclick="toggleSources(\''+id+'\')">▸ ' + data.sources.length + ' source(s)</button>';
      metaHtml += '<div class="src-list" id="'+id+'">';
      data.sources.forEach(s => {
        metaHtml += '<div class="src-item">';
        if (s.ticker) metaHtml += '<span class="tag">' + s.ticker + '</span>';
        if (s.section_name) metaHtml += s.section_name;
        if (s.filing_type) metaHtml += ' · ' + s.filing_type;
        metaHtml += '<span class="snippet">' + (s.content || '').substring(0, 180) + '…</span></div>';
      });
      metaHtml += '</div>';
    }
    meta.innerHTML = metaHtml;
  } catch (e) {
    answer.textContent = '> network error — is the server running?';
  } finally {
    btn.disabled = false;
    btn.textContent = 'RUN';
  }
}

function toggleSources(id) {
  document.getElementById(id).classList.toggle('open');
}
