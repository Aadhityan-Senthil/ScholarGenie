"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import axios from "axios"
import {
  Search, FileText, Network, TrendingUp, BookOpen, Download,
  Sparkles, Loader2, AlertCircle, ChevronDown, ChevronUp,
  RefreshCw, ExternalLink, Brain, Zap, ArrowLeft, Star,
  Calendar, Quote, MessageSquare, Send, Lightbulb, BookMarked,
  Bot, CheckCircle, FlaskConical, BarChart3, Cpu, Trash2,
  Copy, Link2, Layers, ArrowUpRight
} from "lucide-react"
import Link from "next/link"

// ─── Types ─────────────────────────────────────────────────────────────────────
interface Paper {
  id: string; title: string; authors: string[]; abstract: string
  year: number; url: string; source: string; citations: number
  similarity?: number
}
interface Summary {
  tldr: string; short: string; full: string
  keypoints: string[]; ai_powered?: boolean
}
interface Gap {
  gap_id: string; type: string; title: string; description: string
  confidence: number; impact: string; related_papers: string[]; ai_powered?: boolean
}
interface GraphNode {
  id: string; label: string; type: string; year: number; concepts: string[]
}
interface GraphData {
  nodes: GraphNode[]
  edges: { source: string; target: string; label: string }[]
  shared_concepts: { concept: string; papers: string[]; count: number }[]
}
interface Idea {
  title: string; hypothesis: string; methodology: string
  novelty: string; impact: string; difficulty: string
}
interface ChatMsg { role: "user" | "assistant"; content: string }
interface TrendData {
  years: number[]; counts: number[]
  top_concepts_by_era: Record<string, string[]>
  velocity: Record<string, number>
  rising_topics: { concept: string; early: number; recent: number; growth: number }[]
  year_range?: [number, number]; total_papers?: number
}
interface SimilarPaper extends Paper { similarity: number }

// ─── Interactive Force-Directed Graph ───────────────────────────────────────────
function ForceGraph({
  nodes, edges, backend,
}: {
  nodes: GraphNode[]
  edges: { source: string; target: string; label: string }[]
  backend: string
}) {
  const W = 960, H = 480
  const svgRef   = useRef<SVGSVGElement>(null)
  const posRef   = useRef<Record<string, { x: number; y: number }>>({})
  const velRef   = useRef<Record<string, { vx: number; vy: number }>>({})
  const edgesRef = useRef(edges)
  const animRef  = useRef<number>()
  const dragRef  = useRef<string | null>(null)
  const panRef   = useRef<{ sx: number; sy: number; tx: number; ty: number } | null>(null)
  const tRef     = useRef({ x: 0, y: 0, s: 1 })

  const [tick,     setTick]     = useState(0)
  const [hovered,  setHovered]  = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [tfm,      setTfm]      = useState({ x: 0, y: 0, s: 1 })

  edgesRef.current = edges

  const runSim = (maxIter: number) => {
    if (animRef.current) cancelAnimationFrame(animRef.current)
    let iter = 0
    const step = () => {
      if (iter++ > maxIter) return
      const p = posRef.current, v = velRef.current
      const ids = Object.keys(p)
      // repulsion
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
          const a = ids[i], b = ids[j]
          const dx = p[b].x - p[a].x, dy = p[b].y - p[a].y
          const d = Math.sqrt(dx * dx + dy * dy) || 1
          const f = 5000 / (d * d)
          v[a].vx -= f * dx / d; v[a].vy -= f * dy / d
          v[b].vx += f * dx / d; v[b].vy += f * dy / d
        }
      }
      // springs
      edgesRef.current.forEach(e => {
        const s = p[e.source], t = p[e.target]
        if (!s || !t) return
        const dx = t.x - s.x, dy = t.y - s.y
        const d = Math.sqrt(dx * dx + dy * dy) || 1
        const f = (d - 160) * 0.04
        v[e.source].vx += f * dx / d; v[e.source].vy += f * dy / d
        v[e.target].vx -= f * dx / d; v[e.target].vy -= f * dy / d
      })
      // gravity + integrate
      ids.forEach(id => {
        if (dragRef.current === id) return
        v[id].vx += (W / 2 - p[id].x) * 0.003
        v[id].vy += (H / 2 - p[id].y) * 0.003
        v[id].vx *= 0.85; v[id].vy *= 0.85
        p[id].x = Math.max(25, Math.min(W - 25, p[id].x + v[id].vx))
        p[id].y = Math.max(25, Math.min(H - 25, p[id].y + v[id].vy))
      })
      setTick(t => t + 1)
      animRef.current = requestAnimationFrame(step)
    }
    animRef.current = requestAnimationFrame(step)
  }

  useEffect(() => {
    const pos: Record<string, { x: number; y: number }> = {}
    const vel: Record<string, { vx: number; vy: number }> = {}
    nodes.forEach((n, i) => {
      const a = (i / Math.max(nodes.length, 1)) * 2 * Math.PI - Math.PI / 2
      const r = Math.min(200, 80 + nodes.length * 6)
      pos[n.id] = {
        x: W / 2 + r * Math.cos(a) + (Math.random() - .5) * 15,
        y: H / 2 + r * Math.sin(a) + (Math.random() - .5) * 15,
      }
      vel[n.id] = { vx: 0, vy: 0 }
    })
    posRef.current = pos; velRef.current = vel
    setTick(t => t + 1)
    runSim(300)
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes.length])

  const svgCoord = (cx: number, cy: number) => {
    const rect = svgRef.current!.getBoundingClientRect()
    const t = tRef.current
    return { x: (cx - rect.left - t.x) / t.s, y: (cy - rect.top - t.y) / t.s }
  }

  const onNodeDown = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    dragRef.current = id
    if (animRef.current) cancelAnimationFrame(animRef.current)
  }
  const onBgDown = (e: React.MouseEvent) => {
    panRef.current = { sx: e.clientX, sy: e.clientY, tx: tRef.current.x, ty: tRef.current.y }
  }
  const onMove = (e: React.MouseEvent) => {
    if (dragRef.current) {
      const c = svgCoord(e.clientX, e.clientY)
      posRef.current[dragRef.current] = { x: c.x, y: c.y }
      setTick(t => t + 1)
    } else if (panRef.current) {
      const dx = e.clientX - panRef.current.sx, dy = e.clientY - panRef.current.sy
      const nt = { ...tRef.current, x: panRef.current.tx + dx, y: panRef.current.ty + dy }
      tRef.current = nt; setTfm({ ...nt })
    }
  }
  const onUp = () => {
    if (dragRef.current) { dragRef.current = null; runSim(150) }
    panRef.current = null
  }
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const f = e.deltaY < 0 ? 1.1 : 0.9
    const rect = svgRef.current!.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    const t = tRef.current
    const ns = Math.max(0.2, Math.min(4, t.s * f))
    const nt = { x: mx - (mx - t.x) * (ns / t.s), y: my - (my - t.y) * (ns / t.s), s: ns }
    tRef.current = nt; setTfm({ ...nt })
  }

  const pos = posRef.current
  const activeN = (hovered ? nodes.find(n => n.id === hovered) : null) ||
                  (selected ? nodes.find(n => n.id === selected) : null)

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Network className="h-4 w-4 text-purple-400" />Graph Visualization
          <span className="text-xs text-gray-500">({nodes.length} nodes · {edges.length} edges)</span>
        </h3>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-600 hidden sm:block">Drag · Scroll zoom · Click to inspect</span>
          {backend === "neo4j" ? (
            <span className="text-xs px-2 py-0.5 rounded-full bg-green-900/40 text-green-400 border border-green-800/40 flex items-center gap-1">
              <CheckCircle className="h-3 w-3" />Neo4j
            </span>
          ) : (
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-gray-500">In-memory</span>
          )}
        </div>
      </div>

      <div className="relative rounded-xl overflow-hidden" style={{ height: 420, background: "#050a18" }}>
        <svg ref={svgRef} width="100%" height="100%"
          viewBox={`0 0 ${W} ${H}`}
          className="select-none"
          style={{ cursor: panRef.current ? "grabbing" : "grab" }}
          onMouseDown={onBgDown} onMouseMove={onMove}
          onMouseUp={onUp} onMouseLeave={onUp}
          onWheel={onWheel}>
          <defs>
            <filter id="glow-b" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="4" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="glow-p" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="4" result="b"/>
              <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <radialGradient id="bggrad" cx="50%" cy="40%" r="60%">
              <stop offset="0%" stopColor="#0d1529"/>
              <stop offset="100%" stopColor="#020817"/>
            </radialGradient>
          </defs>
          <rect width={W} height={H} fill="url(#bggrad)"/>
          <g transform={`translate(${tfm.x},${tfm.y}) scale(${tfm.s})`}>
            {edges.map((e, i) => {
              const s = pos[e.source], t = pos[e.target]
              if (!s || !t) return null
              const hi = hovered === e.source || hovered === e.target ||
                         selected === e.source || selected === e.target
              return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                stroke={hi ? "#7c3aed" : "#1e3a5f"}
                strokeWidth={hi ? 2.5 : 1}
                strokeOpacity={hi ? 1 : 0.4}/>
            })}
            {nodes.map(n => {
              const p = pos[n.id]; if (!p) return null
              const cit = (n as any).citations || 0
              const r = Math.min(20, Math.max(8, 8 + Math.log1p(cit) * 1.8))
              const isArx = (n as any).source === "arXiv"
              const fill = isArx ? "#3b82f6" : "#a855f7"
              const glowC = isArx ? "#93c5fd" : "#d8b4fe"
              const isHov = hovered === n.id
              const isSel = selected === n.id
              const connected = hovered
                ? edges.some(e => (e.source === hovered && e.target === n.id) || (e.target === hovered && e.source === n.id))
                : false
              const dim = !!hovered && !isHov && !connected
              return (
                <g key={n.id} style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHovered(n.id)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => setSelected(isSel ? null : n.id)}
                  onMouseDown={ev => onNodeDown(ev, n.id)}>
                  {(isHov || isSel) && (
                    <circle cx={p.x} cy={p.y} r={r + 10} fill={fill} fillOpacity="0.12"/>
                  )}
                  <circle cx={p.x} cy={p.y} r={isHov || isSel ? r + 3 : r}
                    fill={fill}
                    fillOpacity={dim ? 0.15 : isHov || isSel ? 1 : 0.78}
                    stroke={isSel ? "#fff" : isHov ? glowC : "transparent"}
                    strokeWidth={isSel ? 2.5 : 1.5}
                    filter={isHov || isSel ? (isArx ? "url(#glow-b)" : "url(#glow-p)") : undefined}/>
                  <circle cx={p.x - r * 0.28} cy={p.y - r * 0.28} r={r * 0.22}
                    fill="#fff" fillOpacity={dim ? 0.02 : 0.28}
                    style={{ pointerEvents: "none" }}/>
                  {(isHov || isSel) && (
                    <text x={p.x} y={p.y + r + 15} textAnchor="middle"
                      fill="white" fontSize="10" fontWeight="700"
                      style={{ pointerEvents: "none", filter: "drop-shadow(0 1px 5px #000)" }}>
                      {n.label.length > 40 ? n.label.substring(0, 40) + "…" : n.label}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        </svg>
        {activeN && (
          <div className="absolute bottom-3 left-3 right-3 pointer-events-none">
            <div className="bg-slate-950/95 backdrop-blur border border-purple-900/40 rounded-xl p-3">
              <p className="text-sm font-semibold text-white leading-snug mb-1.5">{activeN.label}</p>
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${(activeN as any).source === "arXiv" ? "bg-blue-900/50 text-blue-300" : "bg-purple-900/50 text-purple-300"}`}>
                  {(activeN as any).source || "Unknown"}
                </span>
                {activeN.year && <span className="text-xs text-gray-500">{activeN.year}</span>}
                <span className="text-xs text-gray-500">{(activeN as any).citations || 0} citations</span>
                {activeN.concepts?.slice(0, 4).map((c, i) => (
                  <span key={i} className="text-xs px-1.5 py-0.5 rounded bg-slate-800 text-gray-400">{c}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block"/>arXiv</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-purple-500 inline-block"/>Semantic Scholar</span>
        <span className="text-gray-600 ml-auto">Drag nodes · Scroll to zoom · Click to pin info</span>
      </div>
    </div>
  )
}

// ─── Main Component ─────────────────────────────────────────────────────────────
export default function AppPage() {
  const [activeTab, setActiveTab]           = useState("search")
  const [query, setQuery]                   = useState("")
  const [maxResults, setMaxResults]         = useState(10)
  const [searching, setSearching]           = useState(false)
  const [results, setResults]               = useState<Paper[]>([])
  const [searchError, setSearchError]       = useState("")
  const [semanticMode, setSemanticMode]     = useState(false)
  const [library, setLibrary]               = useState<Paper[]>([])
  const [loadingLibrary, setLoadingLibrary] = useState(false)
  const [deletingPaper, setDeletingPaper]   = useState<Record<string, boolean>>({})
  const [clearingLib, setClearingLib]       = useState(false)
  const [summaries, setSummaries]           = useState<Record<string, Summary>>({})
  const [summarizing, setSummarizing]       = useState<Record<string, boolean>>({})
  const [presenting, setPresenting]         = useState<Record<string, boolean>>({})
  const [downloadLinks, setDownloadLinks]   = useState<Record<string, string>>({})
  const [bibtex, setBibtex]                 = useState<Record<string, string>>({})
  const [similar, setSimilar]               = useState<Record<string, SimilarPaper[]>>({})
  const [loadingSimilar, setLoadingSimilar] = useState<Record<string, boolean>>({})
  const [graphData, setGraphData]           = useState<GraphData | null>(null)
  const [buildingGraph, setBuildingGraph]   = useState(false)
  const [graphBackend, setGraphBackend]     = useState("memory")
  const [gaps, setGaps]                     = useState<Gap[]>([])
  const [discoveringGaps, setDiscoveringGaps] = useState(false)
  const [gapsAiPowered, setGapsAiPowered]   = useState(false)
  const [trendData, setTrendData]           = useState<TrendData | null>(null)
  const [loadingTrends, setLoadingTrends]   = useState(false)
  const [stats, setStats]                   = useState<{ total_papers: number; total_searches: number; total_summaries: number } | null>(null)
  const [aiAvailable, setAiAvailable]       = useState(false)
  const [copied, setCopied]                 = useState<Record<string, boolean>>({})

  // Chat state
  const [chatMessages, setChatMessages]     = useState<ChatMsg[]>([])
  const [chatInput, setChatInput]           = useState("")
  const [chatLoading, setChatLoading]       = useState(false)
  const chatEndRef                          = useRef<HTMLDivElement>(null)

  // Literature Review state
  const [reviewTopic, setReviewTopic]       = useState("")
  const [review, setReview]                 = useState("")
  const [reviewLoading, setReviewLoading]   = useState(false)
  const [reviewWordCount, setReviewWordCount] = useState(0)

  // Research Ideas state
  const [ideasTopic, setIdeasTopic]         = useState("")
  const [ideas, setIdeas]                   = useState<Idea[]>([])
  const [ideasLoading, setIdeasLoading]     = useState(false)
  const [ideasError, setIdeasError]         = useState("")

  // Agent Pipeline state
  const [pipelineQuery, setPipelineQuery]   = useState("")
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineResult, setPipelineResult] = useState<{
    status: string; query: string; papers_found: number;
    gaps_identified: number; agent_output: string; message: string;
    papers: { id: string; title: string; year: number | string; source: string; citations: number; url: string; authors: string[] }[];
    summaries: Record<string, { tldr: string; keypoints: string[] }>;
    gaps: { type: string; title: string; description: string; confidence: number }[];
    presentation_id: string | null;
  } | null>(null)
  const [pipelineError, setPipelineError]   = useState("")
  const [pipelineStep, setPipelineStep]     = useState(0)

  useEffect(() => {
    if (typeof window !== "undefined") {
      const q = new URLSearchParams(window.location.search).get("q")
      if (q) { setQuery(q); performSearch(q, 10, false) }
    }
    fetchStats()
    checkAiStatus()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }) }, [chatMessages])

  const fetchStats = async () => {
    try { const r = await axios.get("/api/stats"); setStats(r.data) } catch {}
  }

  const checkAiStatus = async () => {
    try { const r = await axios.get("/api/ai-status"); setAiAvailable(r.data.ai_available) } catch {}
  }

  const performSearch = async (q: string, n: number, semantic: boolean) => {
    if (!q.trim()) return
    setSearching(true); setSearchError(""); setResults([])
    try {
      let papers: Paper[]
      if (semantic) {
        const r = await axios.post("/api/semantic-search", { query: q.trim(), top_k: n })
        papers = r.data.papers || []
      } else {
        const r = await axios.post("/api/search", { query: q.trim(), max_results: n })
        papers = r.data.papers || []
      }
      setResults(papers)
      fetchStats()
    } catch (e: any) {
      setSearchError(e.response?.data?.detail || "Search failed. Make sure the API server is running: python api.py")
    } finally { setSearching(false) }
  }

  const handleSearch = () => performSearch(query, maxResults, semanticMode)

  const loadLibrary = useCallback(async () => {
    setLoadingLibrary(true)
    try { const r = await axios.get("/api/papers"); setLibrary(r.data.papers || []) }
    catch {} finally { setLoadingLibrary(false) }
  }, [])

  const summarize = async (paperId: string) => {
    setSummarizing(p => ({ ...p, [paperId]: true }))
    try {
      const r = await axios.post("/api/summarize", { paper_id: paperId })
      setSummaries(p => ({ ...p, [paperId]: r.data.summary }))
    } catch {} finally { setSummarizing(p => ({ ...p, [paperId]: false })) }
  }

  const generatePresentation = async (paperId: string) => {
    setPresenting(p => ({ ...p, [paperId]: true }))
    try {
      const r = await axios.post("/api/present", { paper_id: paperId })
      setDownloadLinks(p => ({ ...p, [paperId]: r.data.download }))
    } catch {} finally { setPresenting(p => ({ ...p, [paperId]: false })) }
  }

  const fetchBibtex = async (paperId: string) => {
    try {
      const r = await axios.get(`/api/bibtex/${paperId}`)
      setBibtex(p => ({ ...p, [paperId]: r.data.bibtex }))
    } catch {}
  }

  const copyToClipboard = async (text: string, key: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(p => ({ ...p, [key]: true }))
      setTimeout(() => setCopied(p => ({ ...p, [key]: false })), 2000)
    } catch {}
  }

  const fetchSimilar = async (paperId: string) => {
    setLoadingSimilar(p => ({ ...p, [paperId]: true }))
    try {
      const r = await axios.get(`/api/similar/${paperId}?top_k=5`)
      setSimilar(p => ({ ...p, [paperId]: r.data.similar || [] }))
    } catch {} finally { setLoadingSimilar(p => ({ ...p, [paperId]: false })) }
  }

  const deletePaper = async (paperId: string) => {
    setDeletingPaper(p => ({ ...p, [paperId]: true }))
    try {
      await axios.delete(`/api/papers/${paperId}`)
      setLibrary(lib => lib.filter(p => p.id !== paperId))
      fetchStats()
    } catch {} finally { setDeletingPaper(p => ({ ...p, [paperId]: false })) }
  }

  const clearLibrary = async () => {
    if (!confirm("Remove ALL papers from your library? This cannot be undone.")) return
    setClearingLib(true)
    try {
      await axios.delete("/api/papers")
      setLibrary([])
      fetchStats()
    } catch {} finally { setClearingLib(false) }
  }

  const buildGraph = async () => {
    setBuildingGraph(true)
    try {
      const r = await axios.post("/api/knowledge-graph", {})
      setGraphData(r.data.graph)
      setGraphBackend(r.data.backend || "memory")
    } catch {} finally { setBuildingGraph(false) }
  }

  const discoverGaps = async () => {
    setDiscoveringGaps(true)
    try {
      const r = await axios.post("/api/gaps", {})
      setGaps(r.data.gaps || [])
      setGapsAiPowered(!!r.data.ai_powered)
    } catch {} finally { setDiscoveringGaps(false) }
  }

  const loadTrends = async () => {
    setLoadingTrends(true)
    try { const r = await axios.get("/api/trends"); setTrendData(r.data) }
    catch {} finally { setLoadingTrends(false) }
  }

  const sendChat = async () => {
    if (!chatInput.trim() || chatLoading) return
    const userMsg: ChatMsg = { role: "user", content: chatInput.trim() }
    setChatMessages(p => [...p, userMsg])
    setChatInput("")
    setChatLoading(true)
    try {
      const r = await axios.post("/api/chat", {
        message: userMsg.content,
        history: chatMessages.slice(-8).map(m => ({ role: m.role, content: m.content }))
      })
      setChatMessages(p => [...p, { role: "assistant", content: r.data.reply }])
    } catch (e: any) {
      const err = e.response?.data?.detail || "Chat failed."
      setChatMessages(p => [...p, { role: "assistant", content: `⚠️ ${err}` }])
    } finally { setChatLoading(false) }
  }

  const generateReview = async () => {
    if (!reviewTopic.trim()) return
    setReviewLoading(true); setReview("")
    try {
      const r = await axios.post("/api/review", { topic: reviewTopic })
      setReview(r.data.review); setReviewWordCount(r.data.word_count)
    } catch (e: any) {
      setReview(`Error: ${e.response?.data?.detail || "Failed to generate review."}`)
    } finally { setReviewLoading(false) }
  }

  const generateIdeas = async () => {
    if (!ideasTopic.trim()) return
    setIdeasLoading(true); setIdeas([]); setIdeasError("")
    try {
      const r = await axios.post("/api/ideas", { topic: ideasTopic })
      const list = r.data.ideas || []
      setIdeas(list)
      if (list.length === 0) setIdeasError("No ideas returned. Try a more specific topic.")
    } catch (e: any) {
      setIdeasError(e.response?.data?.detail || "Failed to generate ideas. Restart the API server.")
    } finally { setIdeasLoading(false) }
  }

  const runPipeline = async () => {
    if (!pipelineQuery.trim()) return
    setPipelineRunning(true); setPipelineResult(null); setPipelineError(""); setPipelineStep(1)
    try {
      setPipelineStep(1)
      const r = await axios.post("/api/pipeline", { query: pipelineQuery.trim() }, { timeout: 300000 })
      setPipelineResult(r.data)
      setPipelineStep(4)
      loadLibrary()
    } catch (e: any) {
      setPipelineError(e.response?.data?.detail || "Pipeline failed. Make sure GROQ_API_KEY is set in .env")
      setPipelineStep(0)
    } finally { setPipelineRunning(false) }
  }

  const tabs = [
    { id: "search",   label: "Search",          icon: Search },
    { id: "library",  label: "Library",          icon: BookOpen },
    { id: "graph",    label: "Knowledge Graph",  icon: Network },
    { id: "gaps",     label: "Research Gaps",    icon: TrendingUp },
    { id: "trends",   label: "Trends",           icon: BarChart3 },
    { id: "chat",     label: "AI Chat",          icon: MessageSquare },
    { id: "tools",    label: "AI Tools",         icon: Sparkles },
    { id: "pipeline", label: "Agent Pipeline",   icon: Bot },
  ]

  const paperActions = {
    summaries, summarizing, onSummarize: summarize,
    presenting, downloadLinks, onPresent: generatePresentation,
    bibtex, onBibtex: fetchBibtex, copied, onCopy: copyToClipboard,
    similar, loadingSimilar, onSimilar: fetchSimilar,
    deletingPaper, onDelete: deletePaper,
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-800/50 bg-slate-950/80 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-2 hover:opacity-80 transition">
            <ArrowLeft className="h-4 w-4 text-gray-400" />
            <Sparkles className="h-6 w-6 text-purple-500" />
            <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              ScholarGenie
            </span>
          </Link>
          <div className="flex items-center gap-4">
            {aiAvailable && (
              <span className="flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 bg-green-900/40 border border-green-700/50 text-green-300 rounded-full">
                <CheckCircle className="h-3 w-3" /> AI Active
              </span>
            )}
            {stats && (
              <div className="flex items-center gap-4 text-sm text-gray-400">
                <span className="flex items-center gap-1.5">
                  <FileText className="h-3.5 w-3.5" />{stats.total_papers} papers
                </span>
                <span className="flex items-center gap-1.5">
                  <Search className="h-3.5 w-3.5" />{stats.total_searches} searches
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8 max-w-7xl">
        {/* Tabs */}
        <div className="flex gap-1 bg-slate-900/60 border border-slate-800 rounded-2xl p-1.5 mb-8 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id)
                if (tab.id === "library") loadLibrary()
                if (tab.id === "trends") loadTrends()
              }}
              className={`flex-shrink-0 flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? "bg-purple-600 text-white shadow-lg shadow-purple-500/20"
                  : "text-gray-400 hover:text-white hover:bg-slate-800/50"
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span className="hidden sm:inline">{tab.label}</span>
              {(tab.id === "chat" || tab.id === "tools") && aiAvailable && (
                <span className="hidden sm:inline text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded-full">AI</span>
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">

          {/* ── Search ─────────────────────────────────────────────────────── */}
          {activeTab === "search" && (
            <motion.div key="search" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="flex gap-3 mb-4">
                <div className="relative flex-1">
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                  <input
                    type="text" value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && handleSearch()}
                    placeholder="Search for research papers..."
                    className="w-full pl-12 pr-4 py-4 bg-slate-900/80 border border-slate-700 focus:border-purple-500 rounded-xl text-white placeholder-gray-400 outline-none transition text-lg"
                  />
                </div>
                <select value={maxResults} onChange={e => setMaxResults(Number(e.target.value))}
                  className="px-4 bg-slate-900/80 border border-slate-700 rounded-xl text-white outline-none cursor-pointer">
                  {[5,10,20,30].map(n => <option key={n} value={n}>{n} results</option>)}
                </select>
                <button onClick={handleSearch} disabled={searching || !query.trim()}
                  className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-xl font-semibold transition disabled:opacity-50 flex items-center gap-2 whitespace-nowrap">
                  {searching ? <Loader2 className="h-5 w-5 animate-spin" /> : <Zap className="h-5 w-5" />}
                  {searching ? "Searching..." : "Search"}
                </button>
              </div>

              {/* Semantic search toggle */}
              <div className="flex items-center gap-3 mb-6">
                <button
                  onClick={() => setSemanticMode(!semanticMode)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                    semanticMode
                      ? "bg-purple-900/50 border-purple-600 text-purple-300"
                      : "bg-slate-800/50 border-slate-700 text-gray-400 hover:text-white"
                  }`}
                >
                  <Brain className="h-3.5 w-3.5" />
                  {semanticMode ? "Semantic Search ON" : "Semantic Search OFF"}
                </button>
                <span className="text-xs text-gray-600">
                  {semanticMode ? "Searching by meaning within your library" : "Searching arXiv + Semantic Scholar"}
                </span>
              </div>

              {searchError && (
                <div className="flex items-center gap-3 p-4 bg-red-900/30 border border-red-800 rounded-xl mb-6 text-red-300">
                  <AlertCircle className="h-5 w-5 flex-shrink-0" />{searchError}
                </div>
              )}

              {results.length > 0 && (
                <div>
                  <p className="text-gray-400 text-sm mb-4">
                    {results.length} papers {semanticMode && <span className="text-purple-400">(ranked by semantic similarity)</span>}
                  </p>
                  <div className="space-y-4">
                    {results.map((paper, i) => <PaperCard key={paper.id} paper={paper} index={i} {...paperActions} />)}
                  </div>
                </div>
              )}

              {!searching && results.length === 0 && !searchError && (
                <div className="text-center py-24 text-gray-500">
                  <Brain className="h-16 w-16 mx-auto mb-4 opacity-20" />
                  <p className="text-xl font-medium">Enter a topic to discover research papers</p>
                  <p className="text-sm mt-2 text-gray-600">Sources: arXiv · Semantic Scholar</p>
                  <div className="flex items-center justify-center gap-3 mt-6 flex-wrap">
                    {["Transformer models", "Blockchain P2P", "Quantum computing", "CRISPR gene editing"].map(s => (
                      <button key={s} onClick={() => { setQuery(s); performSearch(s, maxResults, semanticMode) }}
                        className="px-4 py-2 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-lg text-sm text-gray-300 transition">{s}</button>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ── Library ────────────────────────────────────────────────────── */}
          {activeTab === "library" && (
            <motion.div key="library" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold">Saved Papers</h2>
                  <p className="text-gray-400 text-sm mt-1">{library.length} papers in library</p>
                </div>
                <div className="flex items-center gap-2">
                  {library.length > 0 && (
                    <>
                      <button onClick={async () => {
                        const r = await axios.get("/api/bibtex")
                        copyToClipboard(r.data.bibtex, "bibtex_all")
                      }}
                        className="flex items-center gap-2 px-4 py-2 bg-blue-900/40 hover:bg-blue-900/60 border border-blue-800 text-blue-300 rounded-lg transition text-sm">
                        <Copy className="h-4 w-4" />
                        {copied["bibtex_all"] ? "Copied!" : "Export BibTeX"}
                      </button>
                      <button onClick={clearLibrary} disabled={clearingLib}
                        className="flex items-center gap-2 px-4 py-2 bg-red-900/30 hover:bg-red-900/50 border border-red-800 text-red-300 rounded-lg transition text-sm disabled:opacity-50">
                        {clearingLib ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                        Clear All
                      </button>
                    </>
                  )}
                  <button onClick={loadLibrary} disabled={loadingLibrary}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition text-sm">
                    <RefreshCw className={`h-4 w-4 ${loadingLibrary ? "animate-spin" : ""}`} />Refresh
                  </button>
                </div>
              </div>
              {loadingLibrary && <div className="text-center py-24 text-gray-400"><Loader2 className="h-10 w-10 mx-auto mb-4 animate-spin" /><p>Loading library...</p></div>}
              {!loadingLibrary && library.length === 0 && (
                <div className="text-center py-24 text-gray-500">
                  <BookOpen className="h-16 w-16 mx-auto mb-4 opacity-20" />
                  <p className="text-xl font-medium">Library is empty</p>
                  <p className="text-sm mt-2 text-gray-600">Search for papers to add them here automatically</p>
                </div>
              )}
              {library.length > 0 && (
                <div className="space-y-4">
                  {library.map((paper, i) => (
                    <PaperCard key={paper.id} paper={paper} index={i} {...paperActions} showDelete />
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {/* ── Knowledge Graph ────────────────────────────────────────────── */}
          {activeTab === "graph" && (
            <motion.div key="graph" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold">Knowledge Graph</h2>
                  <p className="text-gray-400 text-sm mt-1">Discover connections between papers and research concepts</p>
                </div>
                <button onClick={buildGraph} disabled={buildingGraph}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 rounded-xl font-medium transition disabled:opacity-50">
                  {buildingGraph ? <Loader2 className="h-4 w-4 animate-spin" /> : <Network className="h-4 w-4" />}
                  {buildingGraph ? "Building..." : "Build Graph"}
                </button>
              </div>
              {!graphData && !buildingGraph && (
                <div className="text-center py-24 text-gray-500">
                  <Network className="h-16 w-16 mx-auto mb-4 opacity-20" />
                  <p className="text-xl font-medium">Click &quot;Build Graph&quot; to visualize connections</p>
                  <p className="text-sm mt-2 text-gray-600">Requires papers in your library (search first)</p>
                </div>
              )}
              {graphData && (() => {
                const visNodes = graphData.nodes.slice(0, 30)
                const nodeIds = new Set(visNodes.map(n => n.id))
                const visEdges = graphData.edges
                  .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
                  .slice(0, 80)
                return (
                  <div className="space-y-6">
                    <ForceGraph nodes={visNodes} edges={visEdges} backend={graphBackend} />

                    {/* Text panels */}
                    <div className="grid lg:grid-cols-2 gap-6">
                      <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><Zap className="h-5 w-5 text-yellow-400" />Top Shared Concepts</h3>
                        <div className="space-y-2">
                          {graphData.shared_concepts.slice(0,12).map((c, i) => (
                            <div key={i} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                              <span className="font-medium text-purple-300 text-sm">{c.concept}</span>
                              <span className="text-xs text-gray-400 bg-slate-700/50 px-2 py-0.5 rounded-full">{c.count} papers</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="space-y-4">
                        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                          <h3 className="text-lg font-semibold mb-4">Graph Statistics</h3>
                          <div className="grid grid-cols-2 gap-3">
                            <StatBox label="Papers" value={graphData.nodes.length} color="purple" />
                            <StatBox label="Connections" value={graphData.edges.length} color="pink" />
                            <StatBox label="Shared Concepts" value={graphData.shared_concepts.length} color="green" />
                            <StatBox label="Avg Concepts/Paper" value={Math.round(graphData.nodes.reduce((a,n)=>a+n.concepts.length,0)/Math.max(graphData.nodes.length,1))} color="blue" />
                          </div>
                        </div>
                        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                          <h3 className="text-lg font-semibold mb-4">Paper Nodes</h3>
                          <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                            {graphData.nodes.map((node, i) => (
                              <div key={i} className="flex items-start gap-3 p-2.5 bg-slate-800/40 rounded-lg">
                                <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${(node as any).source === "arXiv" ? "bg-blue-400" : "bg-purple-400"}`} />
                                <div className="min-w-0">
                                  <p className="text-sm font-medium text-white truncate">{node.label}</p>
                                  <p className="text-xs text-gray-400 mt-0.5">{node.year} · {node.concepts.slice(0,4).join(", ")}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </motion.div>
          )}

          {/* ── Research Gaps ──────────────────────────────────────────────── */}
          {activeTab === "gaps" && (
            <motion.div key="gaps" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold">Research Gaps</h2>
                  <p className="text-gray-400 text-sm mt-1">
                    {aiAvailable ? "🤖 AI-powered gap discovery (Llama3)" : "Keyword-based gap analysis"}
                  </p>
                </div>
                <button onClick={discoverGaps} disabled={discoveringGaps}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-xl font-medium transition disabled:opacity-50">
                  {discoveringGaps ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
                  {discoveringGaps ? "Analyzing..." : "Discover Gaps"}
                </button>
              </div>

              {!discoveringGaps && gaps.length === 0 && (
                <div className="text-center py-24 text-gray-500">
                  <TrendingUp className="h-16 w-16 mx-auto mb-4 opacity-20" />
                  <p className="text-xl font-medium">Click &quot;Discover Gaps&quot; to find research opportunities</p>
                  <p className="text-sm mt-2 text-gray-600">
                    {aiAvailable ? "AI will analyze your library and find real gaps" : "Analyzes saved papers to identify unexplored areas"}
                  </p>
                </div>
              )}

              {gaps.length > 0 && (
                <div className="space-y-4">
                  {gapsAiPowered && (
                    <div className="flex items-center gap-2 text-xs text-green-400 bg-green-900/20 border border-green-800/50 rounded-lg px-3 py-2">
                      <Bot className="h-3.5 w-3.5" />Powered by Llama 3.3 70B · {gaps.length} gaps identified
                    </div>
                  )}
                  <div className="grid gap-4">
                    {gaps.map((gap, i) => (
                      <motion.div key={gap.gap_id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                        <div className="flex items-start justify-between gap-4 mb-3">
                          <div>
                            <span className={`inline-block text-xs font-semibold px-2.5 py-1 rounded-full mb-2 ${
                              gap.type.includes("Cross") ? "bg-green-900/50 text-green-300 border border-green-800"
                              : gap.type.includes("Temporal") ? "bg-blue-900/50 text-blue-300 border border-blue-800"
                              : gap.type.includes("Method") ? "bg-yellow-900/50 text-yellow-300 border border-yellow-800"
                              : gap.type.includes("Theor") ? "bg-cyan-900/50 text-cyan-300 border border-cyan-800"
                              : "bg-purple-900/50 text-purple-300 border border-purple-800"
                            }`}>{gap.type}</span>
                            <h3 className="text-lg font-semibold text-white">{gap.title}</h3>
                          </div>
                          <div className="text-right flex-shrink-0">
                            <div className={`text-sm font-bold px-3 py-1 rounded-lg ${
                              gap.impact === "Breakthrough" ? "bg-green-900/50 text-green-300"
                              : gap.impact === "High" || gap.impact === "Very High" ? "bg-orange-900/50 text-orange-300"
                              : "bg-slate-800 text-gray-300"
                            }`}>{gap.impact} Impact</div>
                            <div className="text-xs text-gray-500 mt-1">{Math.round(gap.confidence * 100)}% confidence</div>
                          </div>
                        </div>
                        <p className="text-gray-300 text-sm leading-relaxed">{gap.description}</p>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ── Trends ─────────────────────────────────────────────────────── */}
          {activeTab === "trends" && (
            <motion.div key="trends" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold">Publication Trends</h2>
                  <p className="text-gray-400 text-sm mt-1">Topic evolution, velocity, and rising concepts in your library</p>
                </div>
                <button onClick={loadTrends} disabled={loadingTrends}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 rounded-xl font-medium transition disabled:opacity-50">
                  {loadingTrends ? <Loader2 className="h-4 w-4 animate-spin" /> : <BarChart3 className="h-4 w-4" />}
                  {loadingTrends ? "Analyzing..." : "Refresh Trends"}
                </button>
              </div>

              {loadingTrends && <div className="text-center py-24 text-gray-400"><Loader2 className="h-10 w-10 mx-auto mb-4 animate-spin" /><p>Analyzing trends...</p></div>}

              {!loadingTrends && !trendData && (
                <div className="text-center py-24 text-gray-500">
                  <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-20" />
                  <p className="text-xl font-medium">Trends auto-load when you switch to this tab</p>
                  <p className="text-sm mt-2 text-gray-600">Requires papers in your library</p>
                </div>
              )}

              {trendData && (
                <div className="space-y-6">
                  {/* Summary strip */}
                  {trendData.year_range && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <StatBox label="Total Papers" value={trendData.total_papers || 0} color="purple" />
                      <StatBox label="Year Range" value={trendData.year_range[0]} color="blue" />
                      <StatBox label="Latest Year" value={trendData.year_range[1]} color="green" />
                      <StatBox label="Span (years)" value={trendData.year_range[1] - trendData.year_range[0]} color="pink" />
                    </div>
                  )}

                  {/* Publication velocity chart */}
                  {trendData.years.length > 0 && (
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-cyan-400" />Publication Velocity by Year
                      </h3>
                      <div className="flex items-end gap-2 h-40 overflow-x-auto pb-2">
                        {(() => {
                          const max = Math.max(...trendData.counts, 1)
                          return trendData.years.map((yr, i) => {
                            const h = Math.max(8, (trendData.counts[i] / max) * 140)
                            return (
                              <div key={yr} className="flex flex-col items-center gap-1 flex-shrink-0">
                                <span className="text-xs text-cyan-300 font-semibold">{trendData.counts[i]}</span>
                                <div
                                  className="w-10 rounded-t-md bg-gradient-to-t from-blue-700 to-cyan-500 transition-all"
                                  style={{ height: `${h}px` }}
                                />
                                <span className="text-xs text-gray-500">{yr}</span>
                              </div>
                            )
                          })
                        })()}
                      </div>
                    </div>
                  )}

                  {/* Rising topics */}
                  {trendData.rising_topics && trendData.rising_topics.length > 0 && (
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <ArrowUpRight className="h-5 w-5 text-green-400" />Rising Topics
                        <span className="text-xs text-gray-500 font-normal ml-1">(growing from early to recent papers)</span>
                      </h3>
                      <div className="space-y-3">
                        {trendData.rising_topics.map((t, i) => {
                          const maxGrowth = Math.max(...trendData.rising_topics.map(x => x.growth), 1)
                          const barW = Math.max(10, (t.growth / maxGrowth) * 100)
                          return (
                            <div key={i} className="flex items-center gap-3">
                              <span className="text-sm font-medium text-green-300 w-32 flex-shrink-0">{t.concept}</span>
                              <div className="flex-1 h-5 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full bg-gradient-to-r from-green-700 to-emerald-400 transition-all"
                                  style={{ width: `${barW}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-400 w-16 text-right flex-shrink-0">
                                {t.early}→{t.recent} ({t.growth}×)
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* Concepts by era */}
                  {Object.keys(trendData.top_concepts_by_era).length > 0 && (
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <Layers className="h-5 w-5 text-purple-400" />Top Concepts by Era
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        {Object.entries(trendData.top_concepts_by_era).map(([era, concepts]) => (
                          <div key={era} className="bg-slate-800/40 rounded-xl p-4">
                            <h4 className={`text-sm font-semibold mb-3 ${
                              era === "Recent" ? "text-green-400"
                              : era === "Middle" ? "text-yellow-400"
                              : "text-blue-400"
                            }`}>{era} Period</h4>
                            <div className="flex flex-wrap gap-1.5">
                              {concepts.map((c, i) => (
                                <span key={i} className={`text-xs px-2 py-0.5 rounded-full border ${
                                  era === "Recent" ? "bg-green-900/30 text-green-300 border-green-800/50"
                                  : era === "Middle" ? "bg-yellow-900/30 text-yellow-300 border-yellow-800/50"
                                  : "bg-blue-900/30 text-blue-300 border-blue-800/50"
                                }`}>{c}</span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {/* ── AI Chat ────────────────────────────────────────────────────── */}
          {activeTab === "chat" && (
            <motion.div key="chat" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="mb-6">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <MessageSquare className="h-6 w-6 text-purple-400" />Chat with Your Library
                </h2>
                <p className="text-gray-400 text-sm mt-1">
                  {aiAvailable
                    ? "Ask questions about your saved papers — powered by Llama 3 (Groq)"
                    : "⚠️ Add GROQ_API_KEY to .env to enable AI chat"}
                </p>
              </div>

              {!aiAvailable ? (
                <div className="flex flex-col items-center justify-center py-24 text-gray-500">
                  <Bot className="h-16 w-16 mb-4 opacity-20" />
                  <p className="text-xl font-medium">AI not configured</p>
                  <p className="text-sm mt-2 text-center">Add <code className="bg-slate-800 px-2 py-0.5 rounded text-purple-300">GROQ_API_KEY=your_key</code> to your <code className="bg-slate-800 px-2 py-0.5 rounded text-purple-300">.env</code> file and restart the API</p>
                  <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer"
                    className="mt-4 flex items-center gap-2 px-4 py-2 bg-purple-700 hover:bg-purple-600 rounded-lg text-sm text-white transition">
                    <ExternalLink className="h-4 w-4" />Get Free Groq API Key
                  </a>
                </div>
              ) : (
                <div className="flex flex-col h-[600px] bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
                  <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {chatMessages.length === 0 && (
                      <div className="text-center py-12 text-gray-500">
                        <Bot className="h-12 w-12 mx-auto mb-3 opacity-30" />
                        <p className="font-medium">Ask me anything about your research library</p>
                        <div className="grid grid-cols-2 gap-2 mt-4 max-w-md mx-auto">
                          {[
                            "What are the main themes in my library?",
                            "Which papers discuss blockchain security?",
                            "Summarize the research on P2P networks",
                            "What are the limitations mentioned?",
                          ].map(s => (
                            <button key={s} onClick={() => { setChatInput(s) }}
                              className="text-xs text-left px-3 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 rounded-lg text-gray-300 transition">
                              {s}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {chatMessages.map((msg, i) => (
                      <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                          msg.role === "user"
                            ? "bg-purple-600 text-white rounded-br-sm"
                            : "bg-slate-800 text-gray-200 rounded-bl-sm"
                        }`}>
                          {msg.role === "assistant" && (
                            <div className="flex items-center gap-1.5 mb-1.5 text-xs text-purple-400 font-semibold">
                              <Bot className="h-3 w-3" />ScholarGenie AI
                            </div>
                          )}
                          <p className="whitespace-pre-wrap">{msg.content}</p>
                        </div>
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="flex justify-start">
                        <div className="bg-slate-800 rounded-2xl rounded-bl-sm px-4 py-3">
                          <div className="flex items-center gap-2 text-gray-400 text-sm">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />Thinking...
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>
                  <div className="border-t border-slate-800 p-4 flex gap-3">
                    <input
                      type="text" value={chatInput}
                      onChange={e => setChatInput(e.target.value)}
                      onKeyDown={e => e.key === "Enter" && !e.shiftKey && sendChat()}
                      placeholder="Ask about your research library..."
                      className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 focus:border-purple-500 rounded-xl text-white placeholder-gray-500 outline-none transition text-sm"
                      disabled={chatLoading}
                    />
                    <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()}
                      className="px-4 py-3 bg-purple-600 hover:bg-purple-700 rounded-xl transition disabled:opacity-50">
                      <Send className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ── AI Tools ───────────────────────────────────────────────────── */}
          {activeTab === "tools" && (
            <motion.div key="tools" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="mb-8">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <Sparkles className="h-6 w-6 text-purple-400" />AI Research Tools
                </h2>
                <p className="text-gray-400 text-sm mt-1">
                  {aiAvailable ? "Powered by Llama 3.3 70B (Groq)" : "⚠️ Add GROQ_API_KEY to .env to enable"}
                </p>
              </div>

              {!aiAvailable ? (
                <div className="flex flex-col items-center justify-center py-24 text-gray-500">
                  <Cpu className="h-16 w-16 mb-4 opacity-20" />
                  <p className="text-xl font-medium">AI tools not configured</p>
                  <p className="text-sm mt-2">Add <code className="bg-slate-800 px-2 py-0.5 rounded text-purple-300">GROQ_API_KEY</code> to your .env file</p>
                  <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer"
                    className="mt-4 flex items-center gap-2 px-4 py-2 bg-purple-700 hover:bg-purple-600 rounded-lg text-sm text-white transition">
                    <ExternalLink className="h-4 w-4" />Get Free Groq API Key
                  </a>
                </div>
              ) : (
                <div className="grid lg:grid-cols-2 gap-8">
                  {/* Literature Review */}
                  <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 bg-blue-900/40 rounded-lg"><BookMarked className="h-5 w-5 text-blue-400" /></div>
                      <div>
                        <h3 className="text-lg font-semibold">Literature Review</h3>
                        <p className="text-xs text-gray-500">Full academic review from your library</p>
                      </div>
                    </div>
                    <div className="flex gap-2 mb-4">
                      <input type="text" value={reviewTopic}
                        onChange={e => setReviewTopic(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && generateReview()}
                        placeholder="e.g. blockchain peer-to-peer internet"
                        className="flex-1 px-3 py-2.5 bg-slate-800 border border-slate-700 focus:border-blue-500 rounded-lg text-sm text-white placeholder-gray-500 outline-none transition"
                      />
                      <button onClick={generateReview} disabled={reviewLoading || !reviewTopic.trim()}
                        className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition disabled:opacity-50 flex items-center gap-2">
                        {reviewLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <BookMarked className="h-4 w-4" />}
                        {reviewLoading ? "Writing..." : "Generate"}
                      </button>
                    </div>
                    {review && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-green-400 flex items-center gap-1"><CheckCircle className="h-3 w-3" />Generated · {reviewWordCount} words</span>
                          <button onClick={() => copyToClipboard(review, "review")}
                            className="text-xs text-gray-500 hover:text-white transition flex items-center gap-1">
                            <Copy className="h-3 w-3" />{copied["review"] ? "Copied!" : "Copy"}
                          </button>
                        </div>
                        <div className="bg-slate-800/60 rounded-xl p-4 max-h-96 overflow-y-auto">
                          <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{review}</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Research Ideas */}
                  <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 bg-yellow-900/40 rounded-lg"><Lightbulb className="h-5 w-5 text-yellow-400" /></div>
                      <div>
                        <h3 className="text-lg font-semibold">Research Ideas</h3>
                        <p className="text-xs text-gray-500">Novel hypotheses and directions</p>
                      </div>
                    </div>
                    <div className="flex gap-2 mb-4">
                      <input type="text" value={ideasTopic}
                        onChange={e => setIdeasTopic(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && generateIdeas()}
                        placeholder="e.g. decentralized internet security"
                        className="flex-1 px-3 py-2.5 bg-slate-800 border border-slate-700 focus:border-yellow-500 rounded-lg text-sm text-white placeholder-gray-500 outline-none transition"
                      />
                      <button onClick={generateIdeas} disabled={ideasLoading || !ideasTopic.trim()}
                        className="px-4 py-2.5 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-sm font-medium transition disabled:opacity-50 flex items-center gap-2">
                        {ideasLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Lightbulb className="h-4 w-4" />}
                        {ideasLoading ? "Thinking..." : "Generate"}
                      </button>
                    </div>
                    {ideasError && (
                      <div className="flex items-start gap-2 bg-red-900/20 border border-red-800/50 rounded-xl p-3 mb-3">
                        <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0 mt-0.5" />
                        <p className="text-sm text-red-300">{ideasError}</p>
                      </div>
                    )}
                    {ideas.length > 0 && (
                      <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
                        {ideas.map((idea, i) => (
                          <div key={i} className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                            <div className="flex items-start justify-between gap-2 mb-2">
                              <h4 className="text-sm font-semibold text-yellow-300">{idea.title}</h4>
                              <div className="flex gap-1.5 flex-shrink-0">
                                <span className={`text-xs px-2 py-0.5 rounded-full ${
                                  idea.impact === "Breakthrough" ? "bg-green-900/50 text-green-300"
                                  : idea.impact === "Very High" ? "bg-blue-900/50 text-blue-300"
                                  : "bg-orange-900/50 text-orange-300"}`}>{idea.impact}</span>
                                <span className={`text-xs px-2 py-0.5 rounded-full ${
                                  idea.difficulty === "Hard" || idea.difficulty === "Very Hard" ? "bg-red-900/50 text-red-300"
                                  : idea.difficulty === "Moderate" ? "bg-yellow-900/50 text-yellow-300"
                                  : "bg-green-900/50 text-green-300"}`}>{idea.difficulty}</span>
                              </div>
                            </div>
                            {idea.hypothesis && <div className="mb-2"><span className="text-xs font-semibold text-gray-500 uppercase">Hypothesis </span><p className="text-xs text-gray-300 mt-0.5">{idea.hypothesis}</p></div>}
                            {idea.methodology && <div className="mb-2"><span className="text-xs font-semibold text-gray-500 uppercase">Methodology </span><p className="text-xs text-gray-300 mt-0.5">{idea.methodology}</p></div>}
                            {idea.novelty && <div><span className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1"><FlaskConical className="h-3 w-3" />Novelty </span><p className="text-xs text-gray-400 mt-0.5">{idea.novelty}</p></div>}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* ── Agent Pipeline ──────────────────────────────────────────────── */}
          {activeTab === "pipeline" && (
            <motion.div key="pipeline" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="mb-8">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <Bot className="h-6 w-6 text-green-400" />Agent Pipeline
                </h2>
                <p className="text-gray-400 text-sm mt-1">
                  4 AI agents run in sequence: Discover → Analyse → Find Gaps → Present
                </p>
              </div>

              {/* Input */}
              <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 mb-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Zap className="h-5 w-5 text-yellow-400" />Run Full Research Pipeline
                </h3>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={pipelineQuery}
                    onChange={e => setPipelineQuery(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && !pipelineRunning && runPipeline()}
                    placeholder="e.g. semantic search transformer models"
                    className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 focus:border-green-500 rounded-xl text-sm text-white placeholder-gray-500 outline-none transition"
                    disabled={pipelineRunning}
                  />
                  <button
                    onClick={runPipeline}
                    disabled={pipelineRunning || !pipelineQuery.trim()}
                    className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-xl text-sm font-semibold transition disabled:opacity-50 flex items-center gap-2"
                  >
                    {pipelineRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Bot className="h-4 w-4" />}
                    {pipelineRunning ? "Running..." : "Run Pipeline"}
                  </button>
                </div>
              </div>

              {/* Agent Steps */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                {[
                  { step: 1, label: "Research Agent", desc: "Finds 10 papers from arXiv + Semantic Scholar", icon: Search, color: "blue" },
                  { step: 2, label: "Analysis Agent", desc: "Generates AI summaries for each paper", icon: Brain, color: "purple" },
                  { step: 3, label: "GapFinder Agent", desc: "Identifies research gaps using hybrid AI", icon: Lightbulb, color: "yellow" },
                  { step: 4, label: "Presentation Agent", desc: "Generates 14-slide PPTX from findings", icon: FileText, color: "green" },
                ].map(({ step, label, desc, icon: Icon, color }) => (
                  <div key={step} className={`bg-slate-900/50 border rounded-xl p-4 transition ${
                    pipelineRunning && pipelineStep === step ? `border-${color}-500 shadow-lg shadow-${color}-500/10`
                    : pipelineResult && step <= 4 ? "border-green-800/50"
                    : "border-slate-800"
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        pipelineResult ? "bg-green-700 text-white"
                        : pipelineRunning && pipelineStep === step ? `bg-${color}-600 text-white`
                        : "bg-slate-700 text-gray-400"
                      }`}>
                        {pipelineResult ? <CheckCircle className="h-4 w-4" /> : step}
                      </div>
                      <Icon className={`h-4 w-4 ${
                        pipelineRunning && pipelineStep === step ? `text-${color}-400` : "text-gray-500"
                      }`} />
                    </div>
                    <p className="text-sm font-semibold text-white mb-1">{label}</p>
                    <p className="text-xs text-gray-500">{desc}</p>
                    {pipelineRunning && pipelineStep === step && (
                      <div className="mt-2 flex items-center gap-1 text-xs text-green-400">
                        <Loader2 className="h-3 w-3 animate-spin" />Running...
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Error */}
              {pipelineError && (
                <div className="bg-red-900/20 border border-red-800/50 rounded-xl p-4 mb-6 flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-red-300">Pipeline Failed</p>
                    <p className="text-sm text-red-400 mt-1">{pipelineError}</p>
                  </div>
                </div>
              )}

              {/* Results */}
              {pipelineResult && (
                <div className="space-y-4">
                  <div className="bg-green-900/20 border border-green-700/50 rounded-xl p-4 flex items-center gap-3">
                    <CheckCircle className="h-6 w-6 text-green-400 flex-shrink-0" />
                    <div>
                      <p className="font-semibold text-green-300">Pipeline Complete</p>
                      <p className="text-sm text-green-400">{pipelineResult.message}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 text-center">
                      <p className="text-3xl font-bold text-blue-400">{pipelineResult.papers_found}</p>
                      <p className="text-xs text-gray-500 mt-1">Papers Found</p>
                    </div>
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 text-center">
                      <p className="text-3xl font-bold text-yellow-400">{pipelineResult.gaps_identified}</p>
                      <p className="text-xs text-gray-500 mt-1">Gaps Identified</p>
                    </div>
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 text-center">
                      <p className="text-3xl font-bold text-green-400">4</p>
                      <p className="text-xs text-gray-500 mt-1">Agents Completed</p>
                    </div>
                  </div>
                  {/* ── Papers Found ─────────────────────────────────────────── */}
                  {pipelineResult.papers?.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                        <BookOpen className="h-4 w-4 text-blue-400" />
                        Papers Discovered
                        <span className="ml-1 text-xs px-1.5 py-0.5 rounded bg-blue-900/40 text-blue-400">{pipelineResult.papers.length}</span>
                      </h4>
                      <div className="space-y-2 max-h-[480px] overflow-y-auto pr-1">
                        {pipelineResult.papers.map((p, i) => (
                          <div key={p.id} className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 hover:border-slate-700 transition">
                            <div className="flex items-start justify-between gap-3">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap mb-1">
                                  <span className="text-xs text-gray-600 font-mono w-5 text-right flex-shrink-0">#{i + 1}</span>
                                  <span className="text-xs px-1.5 py-0.5 rounded bg-blue-900/40 text-blue-400">{p.source}</span>
                                  <span className="text-xs text-gray-500">{p.year}</span>
                                  {p.citations > 0 && (
                                    <span className="text-xs text-yellow-600">★ {p.citations} citations</span>
                                  )}
                                </div>
                                <p className="text-sm font-semibold text-white leading-snug">{p.title}</p>
                                {p.authors?.length > 0 && (
                                  <p className="text-xs text-gray-500 mt-0.5">{p.authors.slice(0, 3).join(", ")}</p>
                                )}
                                {pipelineResult.summaries?.[p.id]?.tldr && (
                                  <p className="text-xs text-blue-300/80 mt-2 italic border-l-2 border-blue-700/50 pl-2 leading-relaxed">
                                    {pipelineResult.summaries[p.id].tldr}
                                  </p>
                                )}
                              </div>
                              {p.url && (
                                <a href={p.url} target="_blank" rel="noreferrer"
                                  className="text-gray-700 hover:text-blue-400 transition flex-shrink-0 mt-0.5">
                                  <ExternalLink className="h-3.5 w-3.5" />
                                </a>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ── Research Gaps ─────────────────────────────────────────── */}
                  {pipelineResult.gaps?.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                        <Lightbulb className="h-4 w-4 text-yellow-400" />
                        Research Gaps Identified
                        <span className="ml-1 text-xs px-1.5 py-0.5 rounded bg-yellow-900/40 text-yellow-400">{pipelineResult.gaps.length}</span>
                      </h4>
                      <div className="space-y-2">
                        {pipelineResult.gaps.map((g, i) => {
                          const gapStyles: Record<string, { badge: string; border: string }> = {
                            "Underexplored Area":       { badge: "bg-blue-900/40 text-blue-400",    border: "border-blue-800/40" },
                            "Temporal Gap":             { badge: "bg-purple-900/40 text-purple-400", border: "border-purple-800/40" },
                            "Methodological Gap":       { badge: "bg-orange-900/40 text-orange-400", border: "border-orange-800/40" },
                            "Cross-Domain Opportunity": { badge: "bg-teal-900/40 text-teal-400",     border: "border-teal-800/40" },
                          }
                          const gs = gapStyles[g.type] || { badge: "bg-slate-800 text-gray-400", border: "border-slate-700" }
                          return (
                            <div key={i} className={`bg-slate-950/60 border ${gs.border} rounded-xl p-3`}>
                              <div className="flex items-start gap-2">
                                <span className={`text-xs px-1.5 py-0.5 rounded flex-shrink-0 mt-0.5 whitespace-nowrap ${gs.badge}`}>
                                  {g.type}
                                </span>
                                <div className="min-w-0">
                                  <p className="text-sm font-semibold text-white leading-snug">{g.title}</p>
                                  {g.description && (
                                    <p className="text-xs text-gray-400 mt-1 leading-relaxed">{g.description}</p>
                                  )}
                                </div>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* ── Presentation Download ─────────────────────────────────── */}
                  {pipelineResult.presentation_id && (
                    <div className="bg-green-900/10 border border-green-800/40 rounded-xl p-4 flex items-center justify-between gap-4">
                      <div>
                        <p className="text-sm font-semibold text-white">Research Presentation Ready</p>
                        <p className="text-xs text-gray-500 mt-0.5">14-slide PPTX synthesizing all findings</p>
                      </div>
                      <a
                        href={`/api/present/download/${pipelineResult.presentation_id}`}
                        className="px-4 py-2 bg-green-700 hover:bg-green-600 rounded-lg text-sm font-semibold flex items-center gap-2 transition text-white flex-shrink-0"
                      >
                        <Download className="h-4 w-4" />Download PPTX
                      </a>
                    </div>
                  )}

                  <p className="text-sm text-gray-500 text-center">
                    All papers saved to your Library. View them in Library, Gaps, and Knowledge Graph tabs.
                  </p>
                </div>
              )}

              {!pipelineRunning && !pipelineResult && !pipelineError && (
                <div className="flex flex-col items-center justify-center py-20 text-gray-600">
                  <Bot className="h-16 w-16 mb-4 opacity-20" />
                  <p className="text-lg font-medium">Enter a research topic and click Run Pipeline</p>
                  <p className="text-sm mt-2 text-gray-700">The pipeline takes 2 to 5 minutes to complete. Requires GROQ_API_KEY in .env</p>
                </div>
              )}
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  )
}

// ─── Paper Card ─────────────────────────────────────────────────────────────────
function PaperCard({
  paper, index,
  summaries, summarizing, onSummarize,
  presenting, downloadLinks, onPresent,
  bibtex, onBibtex, copied, onCopy,
  similar, loadingSimilar, onSimilar,
  deletingPaper, onDelete,
  showDelete = false,
}: {
  paper: Paper; index: number; showDelete?: boolean
  summaries: Record<string, Summary>; summarizing: Record<string, boolean>; onSummarize: (id: string) => void
  presenting: Record<string, boolean>; downloadLinks: Record<string, string>; onPresent: (id: string) => void
  bibtex: Record<string, string>; onBibtex: (id: string) => void; copied: Record<string, boolean>; onCopy: (text: string, key: string) => void
  similar: Record<string, SimilarPaper[]>; loadingSimilar: Record<string, boolean>; onSimilar: (id: string) => void
  deletingPaper: Record<string, boolean>; onDelete: (id: string) => void
}) {
  const [expanded, setExpanded]       = useState(false)
  const [showBibtex, setShowBibtex]   = useState(false)
  const [showSimilar, setShowSimilar] = useState(false)

  const summary       = summaries[paper.id]
  const isSummarizing = !!summarizing[paper.id]
  const isPresenting  = !!presenting[paper.id]
  const downloadLink  = downloadLinks[paper.id]
  const paperBibtex   = bibtex[paper.id]
  const similarPapers = similar[paper.id]
  const isLoadingSim  = !!loadingSimilar[paper.id]
  const isDeleting    = !!deletingPaper[paper.id]

  const toggleBibtex = () => {
    if (!paperBibtex) onBibtex(paper.id)
    setShowBibtex(!showBibtex)
  }

  const toggleSimilar = () => {
    if (!similarPapers && !isLoadingSim) onSimilar(paper.id)
    setShowSimilar(!showSimilar)
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.04 }}
      className="bg-slate-900/50 border border-slate-800 hover:border-slate-700 rounded-2xl p-6 transition">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="text-xs font-semibold px-2 py-0.5 bg-purple-900/50 text-purple-300 border border-purple-800/50 rounded-full">{paper.source}</span>
            {paper.year > 0 && <span className="flex items-center gap-1 text-xs text-gray-500"><Calendar className="h-3 w-3" />{paper.year}</span>}
            {paper.citations > 0 && <span className="flex items-center gap-1 text-xs text-gray-500"><Quote className="h-3 w-3" />{paper.citations} citations</span>}
            {paper.similarity !== undefined && (
              <span className="text-xs px-2 py-0.5 bg-green-900/30 text-green-300 border border-green-800/40 rounded-full">
                {Math.round(paper.similarity * 100)}% match
              </span>
            )}
          </div>
          <h3 className="text-base font-semibold text-white leading-snug mb-1">{paper.title}</h3>
          {paper.authors.length > 0 && (
            <p className="text-xs text-gray-500">{paper.authors.slice(0,3).join(", ")}{paper.authors.length > 3 ? ` +${paper.authors.length - 3}` : ""}</p>
          )}
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {paper.url && (
            <a href={paper.url} target="_blank" rel="noopener noreferrer"
              className="p-2 text-gray-500 hover:text-white hover:bg-slate-800 rounded-lg transition"><ExternalLink className="h-4 w-4" /></a>
          )}
          {showDelete && (
            <button onClick={() => onDelete(paper.id)} disabled={isDeleting}
              className="p-2 text-gray-600 hover:text-red-400 hover:bg-red-900/20 rounded-lg transition">
              {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            </button>
          )}
          <button onClick={() => setExpanded(!expanded)} className="p-2 text-gray-500 hover:text-white hover:bg-slate-800 rounded-lg transition">
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {!expanded && paper.abstract && (
        <p className="text-xs text-gray-500 mt-2 leading-relaxed line-clamp-2">{paper.abstract}</p>
      )}

      <div className="flex flex-wrap items-center gap-2 mt-4">
        <button onClick={() => onSummarize(paper.id)} disabled={isSummarizing}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-950/50 hover:bg-blue-900/50 border border-blue-900/50 text-blue-300 rounded-lg text-xs font-medium transition disabled:opacity-50">
          {isSummarizing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Brain className="h-3 w-3" />}
          {isSummarizing ? "Summarizing..." : "Summarize"}
        </button>
        <button onClick={() => onPresent(paper.id)} disabled={isPresenting}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-950/50 hover:bg-purple-900/50 border border-purple-900/50 text-purple-300 rounded-lg text-xs font-medium transition disabled:opacity-50">
          {isPresenting ? <Loader2 className="h-3 w-3 animate-spin" /> : <FileText className="h-3 w-3" />}
          {isPresenting ? "Generating..." : "Generate Slides"}
        </button>
        <button onClick={toggleBibtex}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 text-gray-300 rounded-lg text-xs font-medium transition">
          <Link2 className="h-3 w-3" />BibTeX
        </button>
        <button onClick={toggleSimilar} disabled={isLoadingSim}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 text-gray-300 rounded-lg text-xs font-medium transition disabled:opacity-50">
          {isLoadingSim ? <Loader2 className="h-3 w-3 animate-spin" /> : <Layers className="h-3 w-3" />}
          Similar
        </button>
        {downloadLink && (
          <a href={downloadLink}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-950/50 hover:bg-green-900/50 border border-green-900/50 text-green-300 rounded-lg text-xs font-medium transition">
            <Download className="h-3 w-3" />Download PPTX
          </a>
        )}
      </div>

      {/* BibTeX panel */}
      <AnimatePresence>
        {showBibtex && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
            <div className="mt-4 p-4 bg-slate-800/60 rounded-xl border border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-gray-400 uppercase">BibTeX Citation</span>
                {paperBibtex && (
                  <button onClick={() => onCopy(paperBibtex, `bibtex_${paper.id}`)}
                    className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition">
                    <Copy className="h-3 w-3" />{copied[`bibtex_${paper.id}`] ? "Copied!" : "Copy"}
                  </button>
                )}
              </div>
              {paperBibtex ? (
                <pre className="text-xs text-green-300 font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap">{paperBibtex}</pre>
              ) : (
                <div className="flex items-center gap-2 text-gray-500 text-xs"><Loader2 className="h-3 w-3 animate-spin" />Loading...</div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Similar papers panel */}
      <AnimatePresence>
        {showSimilar && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
            <div className="mt-4 p-4 bg-slate-800/40 rounded-xl border border-slate-700/60">
              <span className="text-xs font-semibold text-gray-400 uppercase block mb-3">Similar Papers in Library</span>
              {isLoadingSim ? (
                <div className="flex items-center gap-2 text-gray-500 text-xs"><Loader2 className="h-3 w-3 animate-spin" />Finding similar...</div>
              ) : similarPapers && similarPapers.length > 0 ? (
                <div className="space-y-2">
                  {similarPapers.map((sp, i) => (
                    <div key={i} className="flex items-start justify-between gap-2 p-2.5 bg-slate-900/50 rounded-lg">
                      <div className="min-w-0">
                        <p className="text-xs font-medium text-white truncate">{sp.title}</p>
                        <p className="text-xs text-gray-500 mt-0.5">{sp.year} · {sp.source}</p>
                      </div>
                      <span className="text-xs px-2 py-0.5 bg-green-900/30 text-green-300 rounded-full flex-shrink-0">
                        {Math.round(sp.similarity * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-gray-500">No similar papers found in library yet.</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Full details panel */}
      <AnimatePresence>
        {(expanded || summary) && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
            <div className="mt-4 pt-4 border-t border-slate-800/60 space-y-4">
              {summary ? (
                <>
                  {summary.ai_powered && (
                    <div className="flex items-center gap-1.5 text-xs text-green-400">
                      <CheckCircle className="h-3 w-3" />AI-powered summary (Llama3)
                    </div>
                  )}
                  {summary.tldr && (
                    <div className="p-4 bg-blue-950/30 border border-blue-900/40 rounded-xl">
                      <p className="text-xs font-semibold text-blue-400 uppercase tracking-wide mb-1.5">TL;DR</p>
                      <p className="text-sm text-blue-100 leading-relaxed">{summary.tldr}</p>
                    </div>
                  )}
                  {summary.keypoints.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Key Contributions</p>
                      <ul className="space-y-1.5">
                        {summary.keypoints.map((kp, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                            <Star className="h-3.5 w-3.5 text-yellow-400 mt-0.5 flex-shrink-0" />{kp}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {summary.full && summary.full !== paper.abstract && (
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Full Summary</p>
                      <p className="text-sm text-gray-300 leading-relaxed">{summary.full}</p>
                    </div>
                  )}
                </>
              ) : (
                <p className="text-sm text-gray-400 leading-relaxed">{paper.abstract}</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// ─── Stat Box ──────────────────────────────────────────────────────────────────
function StatBox({ label, value, color }: { label: string; value: number; color: string }) {
  const colors: Record<string, string> = { purple: "text-purple-400", pink: "text-pink-400", green: "text-green-400", blue: "text-blue-400" }
  return (
    <div className="p-4 bg-slate-800/50 rounded-xl text-center">
      <div className={`text-3xl font-bold ${colors[color] || "text-white"}`}>{value}</div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
    </div>
  )
}
