import React, {useEffect, useMemo, useState} from 'react'
import axios from 'axios'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
const SYSTEM_PROMPT =
  'You are a data scientist. Turn the evaluation report into a concise, vivid summary with clear takeaways. Keep it brief and actionable.'

const escapeHtml = (value) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const inlineMarkdown = (value) => {
  let html = value
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  return html
}

const renderMarkdown = (text) => {
  if (!text) return ''
  const lines = escapeHtml(text).split(/\r?\n/)
  let html = ''
  let inList = false
  lines.forEach((line) => {
    if (line.startsWith('- ')) {
      if (!inList) {
        html += '<ul>'
        inList = true
      }
      html += `<li>${inlineMarkdown(line.slice(2))}</li>`
      return
    }
    if (inList) {
      html += '</ul>'
      inList = false
    }
    if (/^###\s+/.test(line)) {
      html += `<h4>${inlineMarkdown(line.replace(/^###\s+/, ''))}</h4>`
      return
    }
    if (/^##\s+/.test(line)) {
      html += `<h3>${inlineMarkdown(line.replace(/^##\s+/, ''))}</h3>`
      return
    }
    if (/^#\s+/.test(line)) {
      html += `<h2>${inlineMarkdown(line.replace(/^#\s+/, ''))}</h2>`
      return
    }
    if (!line.trim()) {
      html += '<br />'
      return
    }
    html += `<p>${inlineMarkdown(line)}</p>`
  })
  if (inList) html += '</ul>'
  return html
}

const toNumber = (value) => {
  if (value === null || value === undefined || value === '') return null
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

const mean = (values) => {
  const valid = values.filter((v) => Number.isFinite(v))
  if (!valid.length) return null
  return valid.reduce((acc, v) => acc + v, 0) / valid.length
}

const maxValue = (values, fallback = 1) => {
  const valid = values.filter((v) => Number.isFinite(v))
  if (!valid.length) return fallback
  return Math.max(...valid, fallback)
}

const formatMetric = (value) => {
  if (!Number.isFinite(value)) return '--'
  return value.toFixed(3)
}

const MetricBars = ({items, accent = 'var(--accent)'}) => {
  if (!items.length) {
    return <div className="empty">Waiting for results...</div>
  }

  return (
    <div className="bar-list">
      {items.map((item) => {
        const pct = item.max > 0 && Number.isFinite(item.value) ? Math.min(item.value / item.max, 1) : 0
        return (
          <div className="bar-row" key={item.label}>
            <span className="bar-label">{item.label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{width: `${pct * 100}%`, background: accent}} />
            </div>
            <span className="bar-value">{formatMetric(item.value)}</span>
          </div>
        )
      })}
    </div>
  )
}

export default function App() {
  const [gtFile, setGtFile] = useState(null)
  const [genFile, setGenFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState('')
  const [combined, setCombined] = useState([])
  const [error, setError] = useState('')
  const [chatInput, setChatInput] = useState('')
  const [chatMessages, setChatMessages] = useState([])
  const [chatLoading, setChatLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!gtFile || !genFile) {
      setError('Please upload two CSV files (ground truth and generated).')
      return
    }
    const fd = new FormData()
    fd.append('ground_truth', gtFile)
    fd.append('generated', genFile)
    setLoading(true)
    try {
      const r = await axios.post(`${BACKEND_URL}/analyze`, fd, {timeout: 120000})
      setReport(r.data.report || '')
      setCombined(r.data.combined_table || [])
    } catch (err) {
      const serverError = err.response?.data?.error
      const rawMessage = err.message || ''
      const isTimeout = err.code === 'ECONNABORTED' || /timeout/i.test(rawMessage)
      if (isTimeout || !err.response) {
        setError('Server is waking up. Please click "Run analysis" again in about 30 seconds.')
      } else {
        setError('Analysis failed: ' + (serverError || rawMessage))
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!report || chatMessages.length) return
    const defaultPrompt =
      `Please rewrite the following evaluation report into a concise, vivid summary with clear takeaways.\n\n${report}`
    setChatInput(defaultPrompt)
  }, [report, chatMessages.length])

  const handleChatSubmit = async (e) => {
    e.preventDefault()
    if (!chatInput.trim()) return
    const nextMessages = [
      ...chatMessages,
      {role: 'user', content: chatInput.trim()},
    ]
    setChatMessages(nextMessages)
    setChatInput('')
    setChatLoading(true)
    try {
      const r = await axios.post(`${BACKEND_URL}/chat`, {
        messages: [{role: 'system', content: SYSTEM_PROMPT}, ...nextMessages],
      })
      const reply = r.data?.reply || 'No response.'
      setChatMessages((prev) => [...prev, {role: 'assistant', content: reply}])
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {role: 'assistant', content: `Error: ${err.response?.data?.error || err.message}`},
      ])
    } finally {
      setChatLoading(false)
    }
  }

  const metrics = useMemo(() => {
    const rows = Array.isArray(combined) ? combined : []
    const tier1 = rows.filter((r) => r.tier === 'Tier 1')
    const tier2 = rows.find((r) => r.tier === 'Tier 2')
    const tier3 = rows.find((r) => r.tier === 'Tier 3')
    const tier4 = rows.filter((r) => r.tier === 'Tier 4')

    const t1TV = tier1.map((r) => toNumber(r.TV))
    const t1W1 = tier1.map((r) => toNumber(r.W1))
    const t1MeanDiff = tier1.map((r) => {
      const val = toNumber(r.MeanDiff)
      return Number.isFinite(val) ? Math.abs(val) : null
    })
    const t1VarRatio = tier1.map((r) => toNumber(r.VarRatio))

    const t4Dcr = tier4.map((r) => toNumber(r.DCR))
    const t4Smr = tier4.map((r) => toNumber(r.SMR))

    const t2Values = tier2 ? [toNumber(tier2.MAE), toNumber(tier2.RMSE), toNumber(tier2.MaxAbs)] : []
    const t3Values = tier3
      ? [toNumber(tier3.energy_distance), toNumber(tier3.mmd_gaussian), toNumber(tier3.c2st_auc)]
      : []

    const variableSet = new Set(
      rows
        .map((r) => r.variable)
        .filter((v) => v && v !== 'GLOBAL')
    )

    return {
      variableCount: variableSet.size,
      tier1: {
        avgTV: mean(t1TV),
        avgW1: mean(t1W1),
        avgMeanDiff: mean(t1MeanDiff),
        avgVarRatio: mean(t1VarRatio),
        maxTV: maxValue(t1TV),
        maxW1: maxValue(t1W1),
        maxMeanDiff: maxValue(t1MeanDiff),
        maxVarRatio: maxValue(t1VarRatio),
      },
      tier2: {
        MAE: tier2 ? toNumber(tier2.MAE) : null,
        RMSE: tier2 ? toNumber(tier2.RMSE) : null,
        MaxAbs: tier2 ? toNumber(tier2.MaxAbs) : null,
        max: maxValue(t2Values),
      },
      tier3: {
        energy: tier3 ? toNumber(tier3.energy_distance) : null,
        mmd: tier3 ? toNumber(tier3.mmd_gaussian) : null,
        auc: tier3 ? toNumber(tier3.c2st_auc) : null,
        max: maxValue(t3Values),
      },
      tier4: {
        avgDCR: mean(t4Dcr),
        avgSMR: mean(t4Smr),
      },
    }
  }, [combined])

  const tier1Bars = [
    {label: 'TV', value: metrics.tier1.avgTV, max: metrics.tier1.maxTV},
    {label: 'W1', value: metrics.tier1.avgW1, max: metrics.tier1.maxW1},
    {label: '|MeanDiff|', value: metrics.tier1.avgMeanDiff, max: metrics.tier1.maxMeanDiff},
    {label: 'VarRatio', value: metrics.tier1.avgVarRatio, max: metrics.tier1.maxVarRatio},
  ]

  const tier2Bars = [
    {label: 'MAE', value: metrics.tier2.MAE, max: metrics.tier2.max},
    {label: 'RMSE', value: metrics.tier2.RMSE, max: metrics.tier2.max},
    {label: 'MaxAbs', value: metrics.tier2.MaxAbs, max: metrics.tier2.max},
  ]

  const tier3Bars = [
    {label: 'Energy', value: metrics.tier3.energy, max: metrics.tier3.max},
    {label: 'MMD', value: metrics.tier3.mmd, max: metrics.tier3.max},
    {label: 'AUC', value: metrics.tier3.auc, max: 1},
  ]

  const tier4Bars = [
    {label: 'DCR', value: metrics.tier4.avgDCR, max: 1},
    {label: 'SMR', value: metrics.tier4.avgSMR, max: 1},
  ]

  const tableColumns = useMemo(() => {
    if (!combined.length) return []
    const keys = new Set()
    combined.forEach((row) => {
      Object.keys(row).forEach((key) => keys.add(key))
    })
    const ordered = ['tier', 'variable']
    const rest = Array.from(keys).filter((k) => !ordered.includes(k)).sort()
    return ordered.concat(rest)
  }, [combined])

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-text">
          <p className="hero-kicker">Survey Evaluation</p>
          <h1>Data-Driven Survey Fidelity Analysis</h1>
          <p className="hero-sub">
            Upload ground-truth and generated survey CSVs to run Tier 1-4 metrics and
            review results on a visual data canvas.
          </p>
        </div>
        <div className="hero-panel">
          <div className="hero-chip">Tier 1 - Distributions</div>
          <div className="hero-chip">Tier 2 - Associations</div>
          <div className="hero-chip">Tier 3 - Multivariate</div>
          <div className="hero-chip">Tier 4 - Inference</div>
        </div>
      </header>

      <main className="layout">
        <div className="side-column">
          <section className="card uploader-card">
            <h2>Upload Data</h2>
            <p className="muted">
              CSV only. Use column names consistent with the demo to improve auto-detection.
            </p>
            <form onSubmit={handleSubmit} className="uploader">
              <label className="file">
                Ground-truth CSV
                <input type="file" accept=".csv" onChange={(e) => setGtFile(e.target.files?.[0])} />
                <span className="file-name">{gtFile?.name || 'No file selected'}</span>
              </label>
              <label className="file">
                Generated CSV
                <input type="file" accept=".csv" onChange={(e) => setGenFile(e.target.files?.[0])} />
                <span className="file-name">{genFile?.name || 'No file selected'}</span>
              </label>

              <button className="btn primary" type="submit" disabled={loading}>
                {loading ? 'Running analysis...' : 'Run analysis'}
              </button>
              {error && <p className="error">{error}</p>}
            </form>
          </section>

          <section className="card info-card">
            <h2>Quick Guide</h2>
            <p className="muted">
              This workflow compares real survey responses against synthetic results to quantify fidelity.
            </p>
            <div className="info-block">
              <h4>Prepare your CSVs</h4>
              <ul>
                <li>Each row represents one respondent.</li>
                <li>Keep column names consistent across both files.</li>
                <li>Ordinal columns should use ordered numeric codes (e.g., 1-5).</li>
              </ul>
            </div>
            <div className="info-block">
              <h4>Read the canvas</h4>
              <ul>
                <li>Tier 1-3 highlight distribution, association, and multivariate fidelity.</li>
                <li>Tier 4 shows inferential alignment (DCR/SMR).</li>
                <li>Use the LLM Summary to turn metrics into actionable insights.</li>
              </ul>
            </div>
          </section>
        </div>

        <section className="card results-card">
          <div className="section-head">
            <h2>Evaluation Canvas</h2>
            <span className="pill">
              Variables: {Number.isFinite(metrics.variableCount) ? metrics.variableCount : '--'}
            </span>
          </div>

          <div className="metrics-board">
            <div className="stat">
              <p>Tier 1 Avg TV</p>
              <strong>{formatMetric(metrics.tier1.avgTV)}</strong>
              <span>Lower is better</span>
            </div>
            <div className="stat">
              <p>Tier 2 RMSE</p>
              <strong>{formatMetric(metrics.tier2.RMSE)}</strong>
              <span>Association error</span>
            </div>
            <div className="stat">
              <p>Tier 3 AUC</p>
              <strong>{formatMetric(metrics.tier3.auc)}</strong>
              <span>Separability</span>
            </div>
            <div className="stat">
              <p>Tier 4 Avg DCR</p>
              <strong>{formatMetric(metrics.tier4.avgDCR)}</strong>
              <span>Directional consistency</span>
            </div>
          </div>

          <div className="figure-grid">
            <figure className="figure-card">
              <figcaption>Tier 1 - Descriptive Similarity</figcaption>
              <MetricBars items={tier1Bars} accent="var(--accent)" />
            </figure>
            <figure className="figure-card">
              <figcaption>Tier 2 - Association Consistency</figcaption>
              <MetricBars items={tier2Bars} accent="var(--accent-2)" />
            </figure>
            <figure className="figure-card">
              <figcaption>Tier 3 - Multivariate Fidelity</figcaption>
              <MetricBars items={tier3Bars} accent="var(--accent-3)" />
            </figure>
            <figure className="figure-card">
              <figcaption>Tier 4 - Inferential Equivalence</figcaption>
              <MetricBars items={tier4Bars} accent="var(--accent-4)" />
            </figure>
          </div>

          <div className="report-block">
            <h3>LLM Summary</h3>
            <div className="chat-panel">
              <div className="chat-log">
                {chatMessages.length ? (
                  chatMessages.map((msg, idx) => (
                    <div className={`chat-message ${msg.role}`} key={`${msg.role}-${idx}`}>
                      <span className="chat-role">{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                      <div
                        className="markdown-content"
                        dangerouslySetInnerHTML={{__html: renderMarkdown(msg.content)}}
                      />
                    </div>
                  ))
                ) : (
                  <div className="empty">Run analysis to generate the default prompt.</div>
                )}
              </div>
              <form className="chat-input" onSubmit={handleChatSubmit}>
                <textarea
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  rows={5}
                  placeholder="Report-based prompt will appear here after analysis."
                />
                <button className="btn primary" type="submit" disabled={chatLoading || !chatInput.trim()}>
                  {chatLoading ? 'Thinking...' : 'Send'}
                </button>
              </form>
            </div>
          </div>

          <div className="table-block">
            <h3>Metrics Table</h3>
            {combined.length ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      {tableColumns.map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {combined.map((row, idx) => (
                      <tr key={`${row.tier}-${row.variable}-${idx}`}>
                        {tableColumns.map((col) => {
                          const value = row[col]
                          const num = toNumber(value)
                          return <td key={col}>{Number.isFinite(num) ? num.toFixed(4) : value ?? '--'}</td>
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty">Upload data to generate the full table.</div>
            )}
          </div>
        </section>
      </main>

      <footer className="footer">CitySurvey Evaluation - Data Canvas</footer>
    </div>
  )
}
