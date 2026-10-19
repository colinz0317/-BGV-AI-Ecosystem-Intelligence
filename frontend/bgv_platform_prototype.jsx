import { useState, useEffect, Fragment } from "react";
import { createPortal } from "react-dom";

// Static region mapping: normalizes raw geography values from backend
// (e.g. "UNITED_STATES", "North America", "france") into the filter buckets
// used by the Strategic Gaps Library.
const REGION_MAPPING = {
  "europe": "Europe", "eu": "Europe", "france": "Europe", "uk": "Europe", "germany": "Europe", "spain": "Europe", "italy": "Europe",
  "us": "United States", "usa": "United States", "north_america": "United States", "north america": "United States", "united states": "United States", "united_states": "United States",
  "india": "India", "japan": "Japan", "israel": "Israel", "middle_east": "Israel",
  "global": "Global", "worldwide": "Global",
  "africa": "Other Regions", "kenya": "Other Regions", "ghana": "Other Regions",
  "asia_pacific": "Other Regions", "thailand": "Other Regions", "singapore": "Other Regions"
};


const BGVLogo = () => (
  <svg width="80" height="35" viewBox="0 0 8192 3571.1" fill="none">
    {/* B */}
    <path d="M0,341.83H1082.51c168.31,0,493.15,0,713.72,136.29C2060.84,646.51,2121,927.22,2121,1095.61c0,108.25-20.07,296.72-168.39,433-20.07,24.05-160.35,152.38-328.82,196.42v8c441.05,80.21,613.43,401,613.43,705.67,0,340.76-200.5,569.29-425,681.53-224.54,116.29-617.4,116.29-729.72,116.29H0ZM593.35,1520.57h433.09c176.43,0,497.13,0,497.13-372.87,0-360.83-304.68-360.83-509.16-360.83H593.35Zm0,1251h421.06c381.89,0,605.38,0,605.38-388.88,0-413-373-413-669.66-413H593.35Z" fill="#424242" />
    {/* V */}
    <path d="M6047.53,334.44l761.91,2285.47L7575.32,334.44H8192L7119.73,3236.66H6438.52L5366.41,334.44Z" fill="#424242" />
    {/* G (C-shape) */}
    <path d="M3963.69,342c-806.93,0-1461.07,647.16-1461.07,1445.39s654.14,1445.46,1461.07,1445.46c604.73,0,1123.56-363.52,1345.59-881.69H4649.53c-163.6,194.72-409.93,318.74-685.84,318.74-492.65,0-892-395.14-892-882.51S3471,905,3963.69,905c273.31,0,517.69,121.74,681.29,313.3h661.78C5083.35,703,4566.23,342,3963.69,342Z" fill="#424242" />
    {/* Blue bar through G */}
    <path d="M5406.23,1551.61H4171.42v471.61H5405a1443.89,1443.89,0,0,0,1.22-471.61Z" fill="#0014ff" />
  </svg>
);

const MicrosoftLogo = ({ s = 18 }) => (
  <svg width={s} height={s} viewBox="0 0 21 21">
    <rect x="1" y="1" width="9" height="9" fill="#f25022" />
    <rect x="11" y="1" width="9" height="9" fill="#7fba00" />
    <rect x="1" y="11" width="9" height="9" fill="#00a4ef" />
    <rect x="11" y="11" width="9" height="9" fill="#ffb900" />
  </svg>
);

const GoogleLogo = ({ s = 18 }) => (
  <svg width={s} height={s} viewBox="0 0 24 24">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4" />
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
  </svg>
);

const NvidiaLogo = ({ s = 20 }) => (
  <svg width={s} height={s} viewBox="0 0 64 64" fill="#76b900">
    <path d="M23.862 23.46v-3.816l1.13-.047c10.46-.33 17.313 8.998 17.313 8.998s-7.396 10.27-15.335 10.27a9.73 9.73 0 0 1-3.086-.495v-11.59c4.075.495 4.9 2.285 7.326 6.36l5.44-4.57s-3.98-5.206-10.67-5.206c-.707-.024-1.413.024-2.12.094m0-12.626v5.7l1.13-.07c14.534-.495 24.026 11.92 24.026 11.92S38.136 41.622 26.806 41.622c-.99 0-1.955-.094-2.92-.26v3.533c.8.094 1.625.165 2.426.165 10.553 0 18.185-5.394 25.58-11.754 1.225.99 6.242 3.368 7.28 4.405-7.02 5.89-23.39 10.623-32.67 10.623a23.24 23.24 0 0 1-2.591-.141v4.97H64v-42.33zm0 27.536v3.015C14.1 39.644 11.4 29.49 11.4 29.49s4.688-5.182 12.46-6.03v3.298h-.024c-4.075-.495-7.28 3.32-7.28 3.32s1.814 6.43 7.302 8.29M6.548 29.067s5.77-8.527 17.337-9.422v-3.11C11.07 17.572 0 28.408 0 28.408s6.266 18.138 23.862 19.787v-3.298c-12.908-1.602-17.313-15.83-17.313-15.83z" />
  </svg>
);

const OpenAILogo = ({ s = 16 }) => (
  <svg width={s} height={s} viewBox="0 0 256 260" fill="#171717">
    <path d="M239.184 106.203a64.716 64.716 0 0 0-5.576-53.103C219.452 28.459 191 15.784 163.213 21.74A65.586 65.586 0 0 0 52.096 45.22a64.716 64.716 0 0 0-43.23 31.36c-14.31 24.602-11.061 55.634 8.033 76.74a64.665 64.665 0 0 0 5.525 53.102c14.174 24.65 42.644 37.324 70.446 31.36a64.72 64.72 0 0 0 48.754 21.744c28.481.025 53.714-18.361 62.414-45.481a64.767 64.767 0 0 0 43.229-31.36c14.137-24.558 10.875-55.423-8.083-76.483Zm-97.56 136.338a48.397 48.397 0 0 1-31.105-11.255l1.535-.87 51.67-29.825a8.595 8.595 0 0 0 4.247-7.367v-72.85l21.845 12.636c.218.111.37.32.409.563v60.367c-.056 26.818-21.783 48.545-48.601 48.601Zm-104.466-44.61a48.345 48.345 0 0 1-5.781-32.589l1.534.921 51.722 29.826a8.339 8.339 0 0 0 8.441 0l63.181-36.425v25.221a.87.87 0 0 1-.358.665l-52.335 30.184c-23.257 13.398-52.97 5.431-66.404-17.803ZM23.549 85.38a48.499 48.499 0 0 1 25.58-21.333v61.39a8.288 8.288 0 0 0 4.195 7.316l62.874 36.272-21.845 12.636a.819.819 0 0 1-.767 0L41.353 151.53c-23.211-13.454-31.171-43.144-17.804-66.405v.256Zm179.466 41.695-63.08-36.63L161.73 77.86a.819.819 0 0 1 .768 0l52.233 30.184a48.6 48.6 0 0 1-7.316 87.635v-61.391a8.544 8.544 0 0 0-4.4-7.213Zm21.742-32.69-1.535-.922-51.619-30.081a8.39 8.39 0 0 0-8.492 0L99.98 99.808V74.587a.716.716 0 0 1 .307-.665l52.233-30.133a48.652 48.652 0 0 1 72.236 50.391v.205ZM88.061 139.097l-21.845-12.585a.87.87 0 0 1-.41-.614V65.685a48.652 48.652 0 0 1 79.757-37.346l-1.535.87-51.67 29.825a8.595 8.595 0 0 0-4.246 7.367l-.051 72.697Zm11.868-25.58L128.067 97.3l28.188 16.218v32.434l-28.086 16.218-28.188-16.218-.052-32.434Z" />
  </svg>
);

const CompanyLogo = ({ name, s }) => {
  if (name === "Microsoft") return <MicrosoftLogo s={s} />;
  if (name === "Google") return <GoogleLogo s={s} />;
  if (name === "NVIDIA") return <NvidiaLogo s={s} />;
  if (name === "OpenAI") return <OpenAILogo s={s} />;
  return null;
};

const DisclaimerNote = ({ children }) => (
  <div style={{
    marginTop: 14,
    maxWidth: 720,
    color: "#64748b",
    fontSize: 11.5,
    lineHeight: 1.55,
  }}>
    <span style={{ color: "#334155" }}>Disclaimer: </span>{children}
  </div>
);

function getScoreColor(score) {
  if (score >= 80) return { accent: "#059669", bg: "#ecfdf5", text: "#065f46", label: "Strong" };
  if (score >= 60) return { accent: "#2563eb", bg: "#eff6ff", text: "#1e40af", label: "Moderate" };
  if (score >= 40) return { accent: "#d97706", bg: "#fffbeb", text: "#92400e", label: "Early" };
  return { accent: "#94a3b8", bg: "#f8fafc", text: "#64748b", label: "Low" };
}

function getHeatmapStyle(score, minScore, maxScore) {
  // If score is exactly 0 (empty / no data), always render as light grey
  if (score === 0) {
    return { cellBg: "rgb(241,245,249)", textColor: "#94a3b8", subColor: "#cbd5e1" };
  }
  // Min-max normalize within the actual data range so full color spectrum is used
  const range = maxScore - minScore;
  const t = range > 0 ? Math.max(0, Math.min(1, (score - minScore) / range)) : 0.5;
  // Interpolate: low → cool blue-grey, high → deep green
  const r = Math.round(219 - t * 189);  // 219 → 30
  const g = Math.round(234 - t * 74);   // 234 → 160
  const b = Math.round(254 - t * 186);  // 254 → 68
  const cellBg = `rgb(${r}, ${g}, ${b})`;
  const textColor = t > 0.5 ? "#ffffff" : "#1e293b";
  const subColor = t > 0.5 ? "rgba(255,255,255,0.75)" : "#94a3b8";
  return { cellBg, textColor, subColor };
}

const SERIF = `'Instrument Serif', Georgia, 'Times New Roman', serif`;
const SANS = `'DM Sans', 'Helvetica Neue', Helvetica, sans-serif`;

export default function BGVPlatform() {
  const [activeTab, setActiveTab] = useState("matrix");
  const [matrixView, setMatrixView] = useState("scope");
  const [selectedCell, setSelectedCell] = useState(null);
  const [gapLibraryRegion, setGapLibraryRegion] = useState("All Regions");
  const [gapLibraryCompany, setGapLibraryCompany] = useState("All");
  const [evidenceViewMode, setEvidenceViewMode] = useState("event"); // "event" | "sub"
  const [portfolioInput, setPortfolioInput] = useState("");
  const [matchResults, setMatchResults] = useState(null);
  const [isMatching, setIsMatching] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [hoveredCell, setHoveredCell] = useState(null);
  const [topK, setTopK] = useState(10);
  const [portfolioDescription, setPortfolioDescription] = useState("");
  const [memo, setMemo] = useState("");
  const [memoMeta, setMemoMeta] = useState(null);
  const [memoItems, setMemoItems] = useState([]);
  const [memoLoading, setMemoLoading] = useState(false);
  const [memoError, setMemoError] = useState("");
  const [memoCache, setMemoCache] = useState({});
  const [gaps, setGaps] = useState([]);
  const [gapsLoading, setGapsLoading] = useState(false);
  const [refreshStatus, setRefreshStatus] = useState(null);
  const [showRefreshDetail, setShowRefreshDetail] = useState(false);

  const [companies, setCompanies] = useState([]);
  const [themesScope, setThemesScope] = useState([]);
  const [themesAgent, setThemesAgent] = useState([]);
  const [scoresScope, setScoresScope] = useState([]);
  const [scoresAgent, setScoresAgent] = useState([]);
  const [portfolioCompanies, setPortfolioCompanies] = useState([]);
  const [evidenceCache, setEvidenceCache] = useState({});
  const [isEvidenceLoading, setIsEvidenceLoading] = useState(false);
  const [expandedSubThemes, setExpandedSubThemes] = useState({});
  const [expandedGapEvidence, setExpandedGapEvidence] = useState({});
  const [expandedMatchEvidence, setExpandedMatchEvidence] = useState({});
  const [scopeIndicesState, setScopeIndicesState] = useState([]);
  const [agentIndicesState, setAgentIndicesState] = useState([]);

  const [issWeights, setIssWeights] = useState({ "Strategic Commitment": 25, "Resourcing & Talent": 20, "Product Shipping": 20, "Commercial Traction": 15, "Partnerships & GTM": 10, "Strategic Consistency": 10 });
  const [dataSources, setDataSources] = useState({ "Earnings Calls": 1.0, "Company Official Site": 0.9, "TechCrunch News": 0.8, "CVC News": 0.8 });

  useEffect(() => {
  if (!showReport || !portfolioInput) return;

  const cacheKey = `${portfolioInput}-${topK}`;
  const cached = memoCache[cacheKey];
    
  if (cached) {
    setMemo(cached.memo || "");
    setMemoMeta(cached.meta || null);
    setMemoItems(cached.items || []);
    setMemoError("");
    setMemoLoading(false);
    return;
  }

  let cancelled = false;
  setMemo("");
  setMemoMeta(null);
  setMemoItems([]);
  setMemoError("");
  setMemoLoading(true);

  (async () => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/recommendation?company=${encodeURIComponent(portfolioInput)}&top_n_gaps=${topK}`);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      if (cancelled) return;

      const items = Array.isArray(data.recommendations)
        ? data.recommendations
        : [{
            memo: data.memo || "",
            gap_title: data.gap_title,
            target_company: data.target_company,
            display_score: data.display_score
          }];

      const first = items[0] || {};
      const meta = {
        gap_title: first.gap_title,
        target_company: first.target_company,
        display_score: first.display_score
      };

      const combinedMemo = items
        .map((item, idx) => {
          const title = item.gap_title || `Recommendation ${idx + 1}`;
          const company = item.target_company || "";
          const score = item.final_score_raw !== undefined && item.final_score_raw !== null
            ? `GMS ${Math.round(Number(item.final_score_raw))}`
            : "GMS N/A";

          return `Recommendation ${idx + 1}: ${company} (${score}) - ${title}\n\n${item.memo || ""}`;
        })
        .join("\n\n---\n\n");

      setMemo(combinedMemo);
      setMemoMeta(meta);
      setMemoItems(items);

      setMemoCache(prev => ({
        ...prev,
        [cacheKey]: {
          memo: combinedMemo,
          meta,
          items
        }
      }));
    } catch (e) {
      if (!cancelled) setMemoError(e.message || "Failed to generate memo.");
    } finally {
      if (!cancelled) setMemoLoading(false);
    }
  })();

  return () => { cancelled = true; };
}, [showReport, portfolioInput, topK, memoCache]);

  useEffect(() => {
    setMounted(true);

    const fetchData = async () => {
      setGapsLoading(true);
      try {
        const [compRes, matrixRes, portRes, setRes, gapsRes, refreshRes] = await Promise.all([
          fetch("http://127.0.0.1:8000/api/companies").catch(() => null),
          fetch("http://127.0.0.1:8000/api/matrix").catch(() => null),
          fetch("http://127.0.0.1:8000/api/portfolio-companies").catch(() => null),
          fetch("http://127.0.0.1:8000/api/settings").catch(() => null),
          fetch("http://127.0.0.1:8000/api/gaps").catch(() => null),
          fetch("http://127.0.0.1:8000/api/refresh-status").catch(() => null),
        ]);

        let scopeIndices = [];
        let agentIndices = [];

        if (compRes && compRes.ok) {
          const compData = await compRes.json();
          setCompanies(compData.companies || []);
          const allThemes = compData.themes || [];

          allThemes.forEach((t, i) => {
            if (t.toLowerCase().includes("agent")) agentIndices.push(i);
            else scopeIndices.push(i);
          });

          setThemesScope(scopeIndices.map(i => allThemes[i]));
          setThemesAgent(agentIndices.map(i => allThemes[i]));
          setScopeIndicesState(scopeIndices);
          setAgentIndicesState(agentIndices);
        }

        if (matrixRes && matrixRes.ok) {
          const matData = await matrixRes.json();
          const allScores = matData.scores || [];
          setScoresScope(allScores.map(row => scopeIndices.map(i => Math.round(row[i] || 0))));
          setScoresAgent(allScores.map(row => agentIndices.map(i => Math.round(row[i] || 0))));
        }

        if (portRes && portRes.ok) {
          const portData = await portRes.json();
          setPortfolioCompanies(portData.companies || []);
        }

        if (setRes && setRes.ok) {
          const setData = await setRes.json();
          if (setData.issWeights) setIssWeights(setData.issWeights);
          if (setData.dataSources) {
            // Backend may return old format — merge with our defaults
            setDataSources(prev => {
              const updated = { ...prev };
              for (const [k, v] of Object.entries(setData.dataSources)) {
                let keyToUpdate = k;
                if (k === "CVC From Google News") keyToUpdate = "CVC News";
                if (keyToUpdate in updated) {
                  const parsed = typeof v === "number" ? v : parseFloat(v);
                  if (!isNaN(parsed)) {
                    updated[keyToUpdate] = parsed;
                  }
                }
              }
              return updated;
            });
          }
        }

        if (gapsRes && gapsRes.ok) {
          const gapsData = await gapsRes.json();
          setGaps(gapsData.gaps || []);
        }

        if (refreshRes && refreshRes.ok) {
          const refreshData = await refreshRes.json();
          setRefreshStatus(refreshData);
        }
      } catch (e) {
        console.error("Failed to fetch initial data", e);
      } finally {
        setGapsLoading(false);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (!mounted || scopeIndicesState.length === 0) return;
    const timer = setTimeout(async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/matrix/recalculate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataSources })
        });
        if (res.ok) {
          const matData = await res.json();
          const allScores = matData.scores || [];
          setScoresScope(allScores.map(row => scopeIndicesState.map(i => Math.round(row[i] || 0))));
          setScoresAgent(allScores.map(row => agentIndicesState.map(i => Math.round(row[i] || 0))));
        }
      } catch (e) {
        console.error("Failed to recalculate matrix", e);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [dataSources, mounted, scopeIndicesState, agentIndicesState]);

  const formatRefreshDate = (isoDate) => {
    if (!isoDate) return "—";
    const [y, m, d] = isoDate.split("-");
    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return `${months[parseInt(m, 10) - 1]} ${parseInt(d, 10)}, ${y}`;
  };

  const totalChunks = null; // chunk display removed; only dates are shown

  const fetchEvidence = async (r, c, company, theme) => {
    const key = `${r}-${c}-${company}-${theme}`;
    if (evidenceCache[key]) return; // already fetched or fetching

    setEvidenceCache(prev => ({ ...prev, [key]: { loading: true, events: [], subThemes: [] } }));
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/evidence?company=${encodeURIComponent(company)}&theme=${encodeURIComponent(theme)}`);
      if (res.ok) {
        const data = await res.json();
        setEvidenceCache(prev => ({ ...prev, [key]: { loading: false, events: data.evidence || [], subThemes: data.subThemes || [] } }));
      }
    } catch (e) {
      console.error(e);
      setEvidenceCache(prev => ({ ...prev, [key]: { loading: false, events: [], subThemes: [] } }));
    }
  };



  // Runs GMS for the selected portfolio company and fetches its description in parallel.
  const handleMatch = async () => {
    if (!portfolioInput) return;
    setIsMatching(true); setMatchResults(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/match?company=${encodeURIComponent(portfolioInput)}&top_k=${topK}`);
      if (res.ok) {
        const data = await res.json();
        setMatchResults(data.matches || []);
      }
      const descRes = await fetch(`http://127.0.0.1:8000/api/portfolio/${encodeURIComponent(portfolioInput)}`);
      if (descRes.ok) {
        const descData = await descRes.json();
        setPortfolioDescription(descData.description || "Portfolio company description not available.");
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsMatching(false);
    }
  };

  const tabs = [
    { id: "matrix", label: "Opportunity Matrix" },
    { id: "gap_library", label: "Strategic Gaps Library" },
    { id: "matcher", label: "Portfolio Matcher" },
    { id: "settings", label: "Settings" },
  ];

  const tabIcons = {
    matrix: <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4"><rect x="1" y="1" width="6" height="6" rx="1.2" /><rect x="9" y="1" width="6" height="6" rx="1.2" /><rect x="1" y="9" width="6" height="6" rx="1.2" /><rect x="9" y="9" width="6" height="6" rx="1.2" /></svg>,
    gap_library: <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4"><path d="M2 4h12M2 8h12M2 12h8" /></svg>,
    matcher: <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4"><circle cx="7" cy="7" r="4.5" /><line x1="10.2" y1="10.2" x2="14" y2="14" /></svg>,
    settings: <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4"><circle cx="8" cy="8" r="2.2" /><path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4" /></svg>,
  };

  const currentThemes = matrixView === "agent" ? themesAgent : themesScope;
  const currentScores = matrixView === "agent" ? scoresAgent : scoresScope;

  const formatTheme = (theme) => {
    if (!theme) return "";
    return theme.split(/\s+|-/).map(word => {
      if (word.toLowerCase() === 'ai') return 'AI';
      return word.charAt(0).toUpperCase() + word.slice(1);
    }).join(' ').replace("Single Agent", "Single-Agent").replace("Multi Agent", "Multi-Agent");
  };

  return (
    <div id="app-root" style={{ display: "flex", height: "100vh", fontFamily: SANS, background: "#fff", color: "#1a1a2e", overflow: "hidden" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; transform: translateX(0); } }
        @keyframes pulse2 { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }
        .hcell:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,0,0,0.07) !important; }
        .hcell { transition: all .25s cubic-bezier(.4,0,.2,1); }
        .mcard:hover { border-color: #c7d0dc !important; }
        .mcard { transition: border-color .2s; }
        .tbtn:hover { background: #f1f5f9 !important; }
        @media print {
          @page { margin: 1cm; }
          html, body {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            background: #fff !important;
            margin: 0 !important;
            padding: 0 !important;
          }
          /* Hide everything at the body level, then show only the report portal */
          body > *:not(.print-backdrop) { display: none !important; }
          .print-backdrop {
            position: static !important;
            width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            background: none !important;
            backdrop-filter: none !important;
            margin: 0 !important;
            padding: 0 !important;
            display: block !important;
          }
          #pdf-report-container {
            position: relative !important;
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            box-shadow: none !important;
            border: none !important;
            border-radius: 0 !important;
            transform: none !important;
            margin: 0 !important;
          }
          .no-print, .no-print * { display: none !important; }
        }
      `}</style>

      {/* ===== SIDEBAR ===== */}
      <div style={{ width: 252, minWidth: 252, background: "#fafbfc", borderRight: "1px solid #eaeff5", display: "flex", flexDirection: "column" }}>
        <div style={{ padding: "28px 24px 20px", borderBottom: "1px solid #eaeff5" }}>
          <div style={{ marginBottom: 14 }}>
            <BGVLogo />
          </div>
          <div style={{ fontFamily: SERIF, fontSize: 20, fontWeight: 400, color: "#0f172a", lineHeight: 1.2, fontStyle: "italic" }}>
            AI Ecosystem Intelligence
          </div>
        </div>

        <div style={{ padding: "14px 24px", borderBottom: "1px solid #eaeff5" }}>
          <div style={{ fontSize: 9.5, color: "#a0aec0", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 3 }}>Last Refresh</div>
          <div style={{ fontSize: 12, color: "#475569", fontWeight: 500 }}>
            {refreshStatus ? formatRefreshDate(refreshStatus.global_last_updated) : "—"}
          </div>
          {refreshStatus && Object.keys(refreshStatus.sources || {}).length > 0 && (
            <div style={{ marginTop: 8 }}>
              <button
                onClick={() => setShowRefreshDetail(v => !v)}
                style={{ fontSize: 9.5, color: "#a0aec0", background: "none", border: "none", cursor: "pointer", padding: 0, fontFamily: "inherit", letterSpacing: "0.04em" }}
              >
                {showRefreshDetail ? "▾ Hide Sources" : "▸ Show Sources"}
              </button>
              {showRefreshDetail && (
                <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
                  {Object.entries(refreshStatus.sources).map(([name, info]) => (
                    <div key={name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 9.5, color: "#94a3b8" }}>
                      <span style={{ flexShrink: 0, maxWidth: 130, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                      <span style={{ fontWeight: 600, color: "#64748b", flexShrink: 0, marginLeft: 4 }}>{formatRefreshDate(info.last_updated)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div style={{ padding: "16px 14px", flex: 1 }}>
          {tabs.map((tab, i) => {
            const active = activeTab === tab.id;
            return (
              <button key={tab.id} className="tbtn" onClick={() => { setActiveTab(tab.id); setSelectedCell(null); }} style={{
                display: "flex", alignItems: "center", gap: 10, width: "100%",
                padding: "11px 14px", marginBottom: 3,
                background: active ? "#fff" : "transparent",
                border: active ? "1px solid #e2e8f0" : "1px solid transparent",
                borderRadius: 10, color: active ? "#0f172a" : "#64748b",
                fontSize: 13, fontWeight: active ? 600 : 400, cursor: "pointer",
                textAlign: "left", fontFamily: "inherit",
                boxShadow: active ? "0 1px 4px rgba(0,0,0,0.04)" : "none",
                animation: mounted ? `slideIn .3s ease ${i * .05}s both` : "none",
              }}>
                <span style={{ color: active ? "#2563eb" : "#94a3b8", display: "flex" }}>{tabIcons[tab.id]}</span>
                {tab.label}
              </button>
            );
          })}
        </div>

        <div style={{ padding: "16px 24px", borderTop: "1px solid #eaeff5", fontSize: 10, color: "#c0c8d4", display: "flex", flexDirection: "column", gap: 4 }}>
          <span>BGV x SCU Practicum 2026</span>
          <span style={{ lineHeight: 1.3 }}>The system is powered by LLMs and can make mistakes. Please double-check responses.</span>
        </div>
      </div>

      {/* ===== MAIN CONTENT ===== */}
      <div id="main-content-area" style={{ flex: 1, overflow: "auto" }}>

        {/* ~~~~~ MATRIX ~~~~~ */}
        {activeTab === "matrix" && (
          <div style={{ padding: "44px 52px", animation: "fadeUp .4s ease" }}>
            <div style={{ marginBottom: 36 }}>
              <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.14em", color: "#a0aec0", textTransform: "uppercase", marginBottom: 8 }}>Investment Intent Scoring</div>
              <h1 style={{ fontFamily: SERIF, fontSize: 32, fontWeight: 400, margin: "0 0 10px", color: "#0f172a", fontStyle: "italic" }}>Opportunity Matrix</h1>
              <p style={{ fontSize: 13.5, color: "#64748b", margin: 0, maxWidth: 580, lineHeight: 1.65 }}>
                Each cell quantifies a company's investment intent (ISS, 0-100) toward a BGV theme. Click any cell to inspect supporting evidence and source data.
              </p>
              <DisclaimerNote>
                ISS is an evidence-weighted directional signal, not a final investment conclusion. Scores depend on extracted events, source reliability, and conservative labeling rules; for agent architecture, Multi-Agent AI is interpreted narrowly as explicit multi-agent orchestration.
              </DisclaimerNote>
            </div>

            <div style={{ display: "flex", background: "#f8fafc", padding: 4, borderRadius: 10, width: "fit-content", border: "1px solid #eaeff5", marginBottom: 20 }}>
              <button onClick={() => { setMatrixView("scope"); setSelectedCell(null); }} style={{
                padding: "6px 16px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
                background: matrixView === "scope" ? "#0f172a" : "transparent",
                color: matrixView === "scope" ? "#fff" : "#64748b",
                border: "none", transition: "all .2s"
              }}>
                Deployment Scope
              </button>
              <button onClick={() => { setMatrixView("agent"); setSelectedCell(null); }} style={{
                padding: "6px 16px", borderRadius: 8, fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
                background: matrixView === "agent" ? "#0f172a" : "transparent",
                color: matrixView === "agent" ? "#fff" : "#64748b",
                border: "none", transition: "all .2s"
              }}>
                Agent Architecture
              </button>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 28 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 11 }}>
                <span style={{ color: "#a0aec0" }}>Low</span>
                <div style={{ width: 180, height: 10, borderRadius: 5, background: "linear-gradient(to right, rgb(240,243,248), rgb(140,203,168), rgb(40,163,88))" }} />
                <span style={{ color: "#a0aec0" }}>Strong</span>
              </div>

              {/* Data Source Priorities */}
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <span style={{ fontSize: 10, fontWeight: 600, color: "#a0aec0", textTransform: "uppercase", letterSpacing: ".06em" }}>Sources Weights</span>
                {Object.entries(dataSources).map(([key, value]) => (
                  <div key={key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontSize: 11, color: "#475569", whiteSpace: "nowrap" }}>{key}</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={value}
                      onChange={e => setDataSources(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                      style={{ width: 40, cursor: "pointer", accentColor: "#0f172a" }}
                    />
                    <span style={{ fontSize: 11, fontWeight: 600, color: "#0f172a", minWidth: 16 }}>{Number(value).toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: `180px repeat(${currentThemes.length}, 270px)`, gap: 16 }}>
              <div />
              {currentThemes.map(t => (
                <div key={t} style={{ padding: "6px 4px 14px", fontSize: 11, fontWeight: 600, color: "#64748b", textAlign: "center" }}>{formatTheme(t)}</div>
              ))}

              {companies.map((company, ri) => (
                <Fragment key={company}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 4px", animation: `slideIn .3s ease ${ri * .06}s both` }}>
                    <div style={{ width: 34, height: 34, borderRadius: 8, background: "#fff", border: "1px solid #eaeff5", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <CompanyLogo name={company} s={18} />
                    </div>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#1e293b" }}>{company}</span>
                  </div>
                  {currentThemes.map((_, ci) => {
                    const score = currentScores[ri]?.[ci] ?? 0;
                    const sc = getScoreColor(score);
                    // Compute min/max across non-zero scores for this view
                    const allNonZero = currentScores.flatMap(row => row.filter(s => s > 0));
                    const minS = allNonZero.length ? Math.min(...allNonZero) : 0;
                    const maxS = allNonZero.length ? Math.max(...allNonZero) : 100;
                    const hm = getHeatmapStyle(score, minS, maxS);
                    const isSel = selectedCell?.r === ri && selectedCell?.c === ci;
                    const isHov = hoveredCell?.r === ri && hoveredCell?.c === ci;
                    return (
                      <button key={`${ri}-${ci}`} className="hcell"
                        onClick={() => {
                          if (!isSel) fetchEvidence(ri, ci, company, currentThemes[ci]);
                          setSelectedCell(isSel ? null : { r: ri, c: ci });
                          setExpandedSubThemes({});
                        }}
                        onMouseEnter={() => setHoveredCell({ r: ri, c: ci })}
                        onMouseLeave={() => setHoveredCell(null)}
                        style={{
                          background: hm.cellBg,
                          border: `1.5px solid ${isSel ? sc.accent : isHov ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.04)"}`,
                          borderRadius: 14, padding: "18px 8px", cursor: "pointer",
                          textAlign: "center", position: "relative", overflow: "hidden", fontFamily: "inherit",
                          animation: `fadeUp .35s ease ${(ri * 4 + ci) * .04}s both`,
                          boxShadow: isSel ? `0 0 0 2px ${sc.accent}40` : "none",
                        }}>
                        <div style={{ fontFamily: SERIF, fontSize: 22, fontWeight: 400, color: hm.textColor, lineHeight: 1 }}>{score}</div>
                        <div style={{ fontSize: 8.5, color: hm.subColor, marginTop: 4, fontWeight: 500, letterSpacing: ".06em", textTransform: "uppercase" }}>{sc.label}</div>
                      </button>
                    );
                  })}
                </Fragment>
              ))}
            </div>

            {selectedCell && (
              <div style={{ marginTop: 32, background: "#fafbfc", border: "1px solid #eaeff5", borderRadius: 16, padding: "28px 32px", maxWidth: 820, animation: "fadeUp .3s ease" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 22 }}>
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                      <div style={{ width: 30, height: 30, borderRadius: 7, background: "#fff", border: "1px solid #eaeff5", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <CompanyLogo name={companies[selectedCell.r]} s={16} />
                      </div>
                      <h3 style={{ fontFamily: SERIF, fontSize: 22, fontWeight: 400, fontStyle: "italic", margin: 0, color: "#0f172a" }}>
                        {companies[selectedCell.r]} - {formatTheme(currentThemes[selectedCell.c])}
                      </h3>
                    </div>
                    <div style={{ fontSize: 12, color: getScoreColor(currentScores[selectedCell.r]?.[selectedCell.c] ?? 0).accent, fontWeight: 600 }}>
                      ISS: {currentScores[selectedCell.r]?.[selectedCell.c] ?? 0} / 100
                    </div>
                  </div>
                  <button onClick={() => setSelectedCell(null)} style={{ background: "#fff", border: "1px solid #e2e8f0", color: "#94a3b8", width: 30, height: 30, borderRadius: 8, cursor: "pointer", fontSize: 14, fontFamily: "inherit", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    x
                  </button>
                </div>

                <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
                  {["event", "sub"].map(mode => (
                    <button key={mode} onClick={() => { setEvidenceViewMode(mode); setExpandedSubThemes({}); }} style={{
                      padding: "6px 14px", borderRadius: 8, fontSize: 11, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
                      background: evidenceViewMode === mode ? "#0f172a" : "#fff",
                      color: evidenceViewMode === mode ? "#fff" : "#64748b",
                      border: "1px solid", borderColor: evidenceViewMode === mode ? "#0f172a" : "#e2e8f0",
                      transition: "all .2s"
                    }}>
                      {mode === "event" ? "Group by Event Type" : "Group by Sub-Theme"}
                    </button>
                  ))}
                </div>

                {evidenceViewMode === "event" ? (
                  evidenceCache[`${selectedCell.r}-${selectedCell.c}-${companies[selectedCell.r]}-${currentThemes[selectedCell.c]}`]?.loading ? (
                    <div style={{ padding: 20, textAlign: "center", color: "#64748b" }}>Loading evidence...</div>
                  ) : (evidenceCache[`${selectedCell.r}-${selectedCell.c}-${companies[selectedCell.r]}-${currentThemes[selectedCell.c]}`]?.events || []).length > 0 ? (
                    evidenceCache[`${selectedCell.r}-${selectedCell.c}-${companies[selectedCell.r]}-${currentThemes[selectedCell.c]}`].events.map((ev, idx) => (
                      <div key={idx} style={{ padding: "16px 20px", background: "#fff", borderRadius: 12, marginBottom: 8, border: "1px solid #eaeff5", animation: `fadeUp .25s ease ${idx * .08}s both` }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                          <span style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#2563eb" }}>{ev.type}</span>
                          <div style={{ display: "flex", gap: 3 }}>
                            {[1, 2, 3, 4, 5].map(i => (
                              <span key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: i <= ev.strength ? "#2563eb" : "#e2e8f0" }} />
                            ))}
                          </div>
                        </div>
                        <div style={{ fontSize: 13, color: "#334155", lineHeight: 1.7, marginBottom: 8 }}>{ev.text}</div>
                        <div style={{ fontSize: 10, color: "#a0aec0" }}>Source: {ev.source}</div>
                      </div>
                    ))
                  ) : (
                    <div style={{ padding: 20, textAlign: "center", color: "#64748b" }}>No evidence found.</div>
                  )
                ) : (
                  (evidenceCache[`${selectedCell.r}-${selectedCell.c}-${companies[selectedCell.r]}-${currentThemes[selectedCell.c]}`]?.subThemes || []).length > 0 ? (
                    evidenceCache[`${selectedCell.r}-${selectedCell.c}-${companies[selectedCell.r]}-${currentThemes[selectedCell.c]}`].subThemes.map((sub, sidx) => {
                      const subKey = `${selectedCell.r}-${selectedCell.c}-${companies[selectedCell.r]}-${currentThemes[selectedCell.c]}-${sub.name}`;
                      const isExpanded = !!expandedSubThemes[subKey];
                      const eventCount = sub.events?.length || 0;
                      return (
                        <div key={sidx} style={{ background: "#fff", borderRadius: 12, marginBottom: 12, border: "1px solid #e2e8f0", animation: `fadeUp .25s ease ${sidx * .08}s both`, overflow: "hidden" }}>
                          <button
                            onClick={() => setExpandedSubThemes(prev => ({ ...prev, [subKey]: !prev[subKey] }))}
                            style={{
                              width: "100%",
                              padding: "16px",
                              background: "#fff",
                              border: "none",
                              cursor: "pointer",
                              fontFamily: "inherit",
                              display: "grid",
                              gridTemplateColumns: "20px minmax(0, 1fr) auto auto",
                              gap: 10,
                              alignItems: "center",
                              textAlign: "left",
                            }}
                          >
                            <span style={{ color: "#94a3b8", fontSize: 14, lineHeight: 1, transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)", transition: "transform .18s ease", display: "inline-block" }}>›</span>
                            <span style={{ fontSize: 13, fontWeight: 700, color: "#0f172a", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{sub.name}</span>
                            <span style={{ fontSize: 10.5, color: "#94a3b8", fontWeight: 600, whiteSpace: "nowrap" }}>{eventCount} events</span>
                            <span style={{ fontSize: 11, fontWeight: 700, color: getScoreColor(sub.iss).accent, background: getScoreColor(sub.iss).bg, padding: "2px 8px", borderRadius: 4, whiteSpace: "nowrap" }}>
                              ISS: {sub.iss}
                            </span>
                          </button>
                          {isExpanded && (
                            <div style={{ padding: "0 16px 16px", borderTop: "1px solid #f1f5f9" }}>
                              {sub.events.map((ev, eidx) => (
                                <div key={eidx} style={{ paddingTop: 12, marginBottom: eidx !== sub.events.length - 1 ? 12 : 0 }}>
                                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                    <span style={{ fontSize: 9.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", color: "#64748b" }}>{ev.type}</span>
                                    <div style={{ display: "flex", gap: 3 }}>
                                      {[1, 2, 3, 4, 5].map(i => <span key={i} style={{ width: 4, height: 4, borderRadius: "50%", background: i <= ev.strength ? "#94a3b8" : "#f1f5f9" }} />)}
                                    </div>
                                  </div>
                                  <div style={{ fontSize: 12.5, color: "#334155", lineHeight: 1.6, marginBottom: 4 }}>{ev.text}</div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <div style={{ padding: 20, textAlign: "center", color: "#64748b" }}>No sub-theme breakdowns found.</div>
                  ))}
              </div>
            )}
          </div>
        )}

        {/* ~~~~~ GAP LIBRARY ~~~~~ */}
        {activeTab === "gap_library" && (
          <div style={{ padding: "44px 52px", animation: "fadeUp .4s ease" }}>
            <div style={{ marginBottom: 36 }}>
              <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.14em", color: "#a0aec0", textTransform: "uppercase", marginBottom: 8 }}>Raw Extraction</div>
              <h1 style={{ fontFamily: SERIF, fontSize: 32, fontWeight: 400, margin: "0 0 10px", color: "#0f172a", fontStyle: "italic" }}>Strategic Gaps Library</h1>
              <p style={{ fontSize: 13.5, color: "#64748b", margin: 0, maxWidth: 600, lineHeight: 1.65 }}>
                Explore automatically extracted strategic capability gaps across top AI hyperscalers. Use the filters below to isolate gaps by target company or operational geography.
              </p>
            </div>

            {/* Filters */}
            <div style={{ display: "flex", gap: 24, marginBottom: 32, paddingBottom: 24, borderBottom: "1px solid #eaeff5" }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 8, textTransform: "uppercase", letterSpacing: ".05em" }}>Filter by Hyperscaler</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {["All", "Google", "Microsoft", "NVIDIA", "OpenAI"].map(comp => (
                    <button key={comp} onClick={() => setGapLibraryCompany(comp)} style={{
                      padding: "6px 14px", borderRadius: 8, fontSize: 12, fontWeight: 500, cursor: "pointer", fontFamily: "inherit", whiteSpace: "nowrap",
                      background: gapLibraryCompany === comp ? "#0f172a" : "#f8fafc",
                      color: gapLibraryCompany === comp ? "#fff" : "#475569",
                      border: "1px solid", borderColor: gapLibraryCompany === comp ? "#0f172a" : "#e2e8f0", transition: "all .2s"
                    }}>
                      {comp}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 8, textTransform: "uppercase", letterSpacing: ".05em" }}>Filter by Region</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {["All Regions", "Europe", "India", "Israel", "Japan", "United States", "Other Regions"].map(reg => (
                    <button key={reg} onClick={() => setGapLibraryRegion(reg)} style={{
                      padding: "6px 14px", borderRadius: 8, fontSize: 12, fontWeight: 500, cursor: "pointer", fontFamily: "inherit", whiteSpace: "nowrap",
                      background: gapLibraryRegion === reg ? "#2563eb" : "#f8fafc",
                      color: gapLibraryRegion === reg ? "#fff" : "#475569",
                      border: "1px solid", borderColor: gapLibraryRegion === reg ? "#2563eb" : "#e2e8f0", transition: "all .2s"
                    }}>
                      {reg}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Gap List */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {gapsLoading && gaps.length === 0 && (
                <div style={{ padding: "40px 0", textAlign: "center", color: "#64748b", fontSize: 13 }}>Loading gaps…</div>
              )}
              {!gapsLoading && gaps.length === 0 && (
                <div style={{ padding: "40px 0", textAlign: "center", color: "#64748b", fontSize: 13 }}>No gaps available. Is the backend running?</div>
              )}
              {gaps.filter(g => {
                if (gapLibraryCompany !== "All" && g.company !== gapLibraryCompany) return false;
                if (gapLibraryRegion !== "All Regions") {
                  const mappedGeos = (g.geography || []).map(geo => REGION_MAPPING[String(geo).toLowerCase()] || "Other Regions");
                  if (!mappedGeos.includes(gapLibraryRegion)) return false;
                }
                return true;
              }).map((g, idx) => {
                const uniqueRegions = Array.from(new Set((g.geography || []).map(geo => REGION_MAPPING[String(geo).toLowerCase()] || "Other Regions")));
                const strength = g.gap_strength || 0;
                const quotes = g.evidence_quotes || [];
                const dates = g.evidence_dates || [];
                const sources = g.source_types || [];
                const isEvidenceOpen = !!expandedGapEvidence[idx];

                const formatSourceLabel = (src) => {
                  const map = {
                    earning_call: "Earnings Call",
                    techCrunch: "TechCrunch",
                    official_website_announcement: "Official Site",
                    CVC_from_google_news: "CVC News",
                    company_announcement: "Company Announcement",
                  };
                  return map[src] || src;
                };

                const formatDate = (iso) => {
                  if (!iso) return "";
                  const [y, m, d] = iso.split("-");
                  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                  return `${months[parseInt(m,10)-1]} ${parseInt(d,10)}, ${y}`;
                };

                return (
                  <div key={idx} style={{ padding: "20px 24px", background: "#fff", borderRadius: 12, border: "1px solid #eaeff5", animation: `fadeUp .25s ease ${idx * .05}s both`, display: "flex", gap: 20 }}>
                    <div style={{ paddingTop: 4 }}>
                      <CompanyLogo name={g.company} s={24} />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10 }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: "#0f172a" }}>{g.company}</span>
                        <span style={{ fontSize: 11, color: "#2563eb", fontWeight: 600, display: "flex", alignItems: "center", gap: 4 }}>
                          {uniqueRegions.join(", ")}
                        </span>
                      </div>
                      {g.gap_title && (
                        <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a", marginBottom: 6, lineHeight: 1.4 }}>
                          {g.gap_title}
                        </div>
                      )}
                      <div style={{ fontSize: 13.5, color: "#334155", lineHeight: 1.6, marginBottom: 12 }}>
                        {g.gap_text}
                      </div>

                      {/* Bottom row: strength dots + evidence toggle */}
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
                          <span style={{ fontSize: 10, color: "#a0aec0", fontWeight: 600, textTransform: "uppercase", marginRight: 4 }}>Gap Strength:</span>
                          {[1, 2, 3, 4, 5].map(i => <span key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: i <= strength ? "#d97706" : "#f1f5f9", display: "inline-block" }} />)}
                        </div>
                        {quotes.length > 0 && (
                          <button
                            onClick={() => setExpandedGapEvidence(prev => ({ ...prev, [idx]: !prev[idx] }))}
                            style={{
                              display: "flex", alignItems: "center", gap: 5,
                              background: "none", border: "none", cursor: "pointer",
                              fontSize: 11, color: isEvidenceOpen ? "#2563eb" : "#94a3b8",
                              fontFamily: "inherit", fontWeight: 600, padding: 0,
                              transition: "color .15s",
                            }}
                          >
                            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8">
                              <path d="M2 4h12M2 8h8M2 12h5" />
                            </svg>
                            {isEvidenceOpen ? "Hide Evidence" : `Show Evidence (${quotes.length})`}
                            <span style={{ transform: isEvidenceOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform .18s", display: "inline-block", fontSize: 10 }}>▾</span>
                          </button>
                        )}
                      </div>

                      {/* Evidence panel */}
                      {isEvidenceOpen && quotes.length > 0 && (
                        <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                          {quotes.map((q, qidx) => (
                            <div key={qidx} style={{
                              background: "#f8fafc", borderRadius: 8,
                              borderLeft: "3px solid #2563eb",
                              padding: "12px 14px",
                            }}>
                              <div style={{ fontSize: 12.5, color: "#1e293b", lineHeight: 1.65, fontStyle: "italic", marginBottom: 8 }}>
                                &ldquo;{q.quote}&rdquo;
                              </div>
                              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                                {sources[qidx] && (
                                  <span style={{ fontSize: 10, fontWeight: 700, color: "#2563eb", textTransform: "uppercase", letterSpacing: ".06em" }}>
                                    {formatSourceLabel(sources[qidx])}
                                  </span>
                                )}
                                {!sources[qidx] && sources[0] && (
                                  <span style={{ fontSize: 10, fontWeight: 700, color: "#2563eb", textTransform: "uppercase", letterSpacing: ".06em" }}>
                                    {formatSourceLabel(sources[0])}
                                  </span>
                                )}
                                {dates[qidx] && (
                                  <span style={{ fontSize: 10, color: "#94a3b8" }}>· {formatDate(dates[qidx])}</span>
                                )}
                                <span style={{ fontSize: 10, color: "#cbd5e1" }}>· {q.chunk_id}</span>
                              </div>
                            </div>
                          ))}
                          {g.gap_strength_reason && (
                            <div style={{ fontSize: 11, color: "#64748b", fontStyle: "italic", paddingLeft: 4 }}>
                              <span style={{ fontWeight: 600, fontStyle: "normal", color: "#94a3b8" }}>Strength rationale: </span>
                              {g.gap_strength_reason}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ~~~~~ MATCHER ~~~~~ */}
        {activeTab === "matcher" && (
          <div style={{ padding: "44px 52px", animation: "fadeUp .4s ease" }}>
            <div style={{ marginBottom: 36 }}>
              <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.14em", color: "#a0aec0", textTransform: "uppercase", marginBottom: 8 }}>Gap Match Scoring</div>
              <h1 style={{ fontFamily: SERIF, fontSize: 32, fontWeight: 400, margin: "0 0 10px", color: "#0f172a", fontStyle: "italic" }}>Portfolio Matcher</h1>
              <p style={{ fontSize: 13.5, color: "#64748b", margin: 0, maxWidth: 600, lineHeight: 1.65 }}>
                Select a portfolio company and run the GMS algorithm to identify the most relevant capability gaps across target companies.
              </p>
              <DisclaimerNote>
                GMS is a screening aid for matching portfolio capabilities to strategic gaps, not a guarantee of strategic fit. The displayed score reflects raw absolute fit; relative rank within each gap may differ, so supporting evidence should be reviewed before action.
              </DisclaimerNote>
            </div>

            <div style={{ display: "flex", gap: 12, marginBottom: 36, alignItems: "center" }}>
              <select value={portfolioInput} onChange={e => { setPortfolioInput(e.target.value); setMatchResults(null); setShowReport(false); setPortfolioDescription(""); }} style={{
                width: 300, padding: "12px 16px", background: "#fafbfc", border: "1px solid #e2e8f0",
                borderRadius: 10, color: "#1e293b", fontSize: 13, fontFamily: "inherit", appearance: "none", cursor: "pointer",
              }}>
                <option value="">Select portfolio company...</option>
                {portfolioCompanies.map(c => <option key={c} value={c}>{c}</option>)}
              </select>

              {/* Top-K selector moved here from Settings */}
              <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 8px" }}>
                <span style={{ fontSize: 11, color: "#64748b", whiteSpace: "nowrap" }}>Top N Gaps</span>
                <input type="range" min="1" max="20" step="1" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} style={{ width: 80, cursor: "pointer", accentColor: "#0f172a" }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: "#0f172a", fontFamily: SERIF, minWidth: 18, textAlign: "center" }}>{topK}</span>
              </div>

              <button onClick={handleMatch} disabled={!portfolioInput || isMatching} style={{
                padding: "12px 30px", background: !portfolioInput || isMatching ? "#f1f5f9" : "#0f172a",
                border: "none", borderRadius: 10, color: !portfolioInput || isMatching ? "#94a3b8" : "#fff",
                fontSize: 13, fontWeight: 600, cursor: !portfolioInput || isMatching ? "not-allowed" : "pointer",
                fontFamily: "inherit", transition: "all .25s",
              }}>
                {isMatching ? "Matching..." : "Start Match"}
              </button>
            </div>

            {isMatching && (
              <div style={{ padding: "56px 0", textAlign: "center" }}>
                <div style={{ width: 30, height: 30, border: "2.5px solid #eaeff5", borderTopColor: "#0f172a", borderRadius: "50%", margin: "0 auto 20px", animation: "spin .8s linear infinite" }} />
                <div style={{ fontSize: 13, color: "#475569" }}>Running GMS against <strong>{portfolioInput}</strong></div>
                <div style={{ fontSize: 11, marginTop: 4, color: "#a0aec0" }}>Semantic retrieval + feature re-ranking</div>
              </div>
            )}

            {matchResults && (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                  <div style={{ fontSize: 10.5, color: "#64748b", letterSpacing: ".04em", textTransform: "uppercase" }}>
                    Top Matches for <span style={{ color: "#0f172a", fontWeight: 700 }}>{portfolioInput}</span>
                  </div>
                  <button onClick={() => setShowReport(true)} style={{
                    padding: "9px 20px", background: "#fff", border: "1px solid #e2e8f0", borderRadius: 10,
                    color: "#475569", fontSize: 12, fontWeight: 500, cursor: "pointer", fontFamily: "inherit",
                    display: "flex", alignItems: "center", gap: 7,
                  }}>
                    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M4 1h8l3 3v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1z" /><path d="M10 1v3h3" /><path d="M6 8h4M6 11h2" /></svg>
                    Generate Report
                  </button>
                </div>

                {matchResults.map((match, idx) => {
                  const sc = getScoreColor(match.gms);
                  const quotes = match.evidence_quotes || [];
                  const dates = match.evidence_dates || [];
                  const sources = match.source_types || [];
                  const isEvidenceOpen = !!expandedMatchEvidence[idx];

                  const formatSourceLabel = (src) => {
                    const map = {
                      earning_call: "Earnings Call",
                      techCrunch: "TechCrunch",
                      official_website_announcement: "Official Site",
                      CVC_from_google_news: "CVC News",
                      company_announcement: "Company Announcement",
                    };
                    return map[src] || src;
                  };

                  const formatDate = (iso) => {
                    if (!iso) return "";
                    const [y, m, d] = iso.split("-");
                    const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                    return `${months[parseInt(m,10)-1]} ${parseInt(d,10)}, ${y}`;
                  };

                  return (
                    <div key={idx} className="mcard" style={{
                      background: "#fff", border: "1px solid #eaeff5", borderRadius: 14,
                      padding: "22px 26px", marginBottom: 10, animation: `fadeUp .3s ease ${idx * .08}s both`,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 15, fontWeight: 600, color: "#0f172a", marginBottom: 6 }}>{match.gap}</div>
                          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "#64748b" }}>
                            <div style={{ width: 20, height: 20, borderRadius: 5, background: "#fafbfc", border: "1px solid #eaeff5", display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <CompanyLogo name={match.company_need} s={12} />
                            </div>
                            {match.company_need}
                          </div>
                        </div>
                        <div style={{ textAlign: "right", flexShrink: 0, marginLeft: 24 }}>
                          <div style={{ fontFamily: SERIF, fontSize: 34, fontWeight: 400, color: sc.accent, lineHeight: 1 }}>{match.gms}</div>
                          <div style={{ fontSize: 9, color: "#a0aec0", textTransform: "uppercase", letterSpacing: ".06em", marginTop: 2 }}>GMS</div>
                        </div>
                      </div>

                      <div style={{ display: "flex", gap: 20, marginBottom: 14, fontSize: 11 }}>
                        {[{ label: "Capability", val: match.capability_match }, { label: "Industry", val: match.industry_match }].filter(m => m.val !== null).map(m => (
                          <div key={m.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ color: "#a0aec0", minWidth: 60 }}>{m.label}</span>
                            <div style={{ width: 64, height: 4, background: "#f1f5f9", borderRadius: 2, overflow: "hidden" }}>
                              <div style={{ width: `${m.val * 100}%`, height: "100%", background: sc.accent, borderRadius: 2, transition: "width .6s ease" }} />
                            </div>
                            <span style={{ color: "#475569", fontWeight: 600 }}>{(m.val * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>

                      {/* Gap summary + evidence toggle row */}
                      <div style={{ padding: "13px 16px", background: "#fafbfc", borderRadius: 8 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 5 }}>
                          <div style={{ fontSize: 10, fontWeight: 700, color: sc.accent, textTransform: "uppercase", letterSpacing: ".05em" }}>Strategic Gap Content</div>
                          {quotes.length > 0 && (
                            <button
                              onClick={() => setExpandedMatchEvidence(prev => ({ ...prev, [idx]: !prev[idx] }))}
                              style={{
                                display: "flex", alignItems: "center", gap: 4,
                                background: "none", border: "none", cursor: "pointer",
                                fontSize: 10, color: isEvidenceOpen ? "#2563eb" : "#94a3b8",
                                fontFamily: "inherit", fontWeight: 600, padding: 0,
                                transition: "color .15s", flexShrink: 0, marginLeft: 12,
                              }}
                            >
                              <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M2 4h12M2 8h8M2 12h5" /></svg>
                              {isEvidenceOpen ? "Hide Evidence" : `Show Evidence (${quotes.length})`}
                              <span style={{ transform: isEvidenceOpen ? "rotate(180deg)" : "rotate(0deg)", transition: "transform .18s", display: "inline-block" }}>▾</span>
                            </button>
                          )}
                        </div>
                        <div style={{ fontSize: 12.5, color: "#475569", lineHeight: 1.65, marginBottom: 5 }}>{match.gap_text || match.evidence}</div>
                        <div style={{ fontSize: 10, color: "#a0aec0" }}>Source: {match.source}</div>
                      </div>

                      {/* Evidence panel */}
                      {isEvidenceOpen && quotes.length > 0 && (
                        <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
                          {quotes.map((q, qidx) => (
                            <div key={qidx} style={{
                              background: "#f8fafc", borderRadius: 8,
                              borderLeft: "3px solid #2563eb",
                              padding: "11px 14px",
                            }}>
                              <div style={{ fontSize: 12.5, color: "#1e293b", lineHeight: 1.65, fontStyle: "italic", marginBottom: 7 }}>
                                &ldquo;{q.quote}&rdquo;
                              </div>
                              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                                {sources[qidx] && (
                                  <span style={{ fontSize: 10, fontWeight: 700, color: "#2563eb", textTransform: "uppercase", letterSpacing: ".06em" }}>
                                    {formatSourceLabel(sources[qidx])}
                                  </span>
                                )}
                                {!sources[qidx] && sources[0] && (
                                  <span style={{ fontSize: 10, fontWeight: 700, color: "#2563eb", textTransform: "uppercase", letterSpacing: ".06em" }}>
                                    {formatSourceLabel(sources[0])}
                                  </span>
                                )}
                                {dates[qidx] && (
                                  <span style={{ fontSize: 10, color: "#94a3b8" }}>· {formatDate(dates[qidx])}</span>
                                )}
                                <span style={{ fontSize: 10, color: "#cbd5e1" }}>· {q.chunk_id}</span>
                              </div>
                            </div>
                          ))}
                          {match.gap_strength_reason && (
                            <div style={{ fontSize: 11, color: "#64748b", fontStyle: "italic", paddingLeft: 4 }}>
                              <span style={{ fontWeight: 600, fontStyle: "normal", color: "#94a3b8" }}>Strength rationale: </span>
                              {match.gap_strength_reason}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* ---- Report Modal (portaled to body so print CSS can cleanly isolate it) ---- */}
            {showReport && createPortal(
              <div className="print-backdrop" onClick={() => setShowReport(false)} style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(15,23,42,.35)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100, fontFamily: SANS, color: "#1a1a2e" }}>
                <div id="pdf-report-container" onClick={e => e.stopPropagation()} style={{ background: "#fff", borderRadius: 16, width: 600, maxHeight: "88vh", overflow: "auto", boxShadow: "0 24px 80px rgba(0,0,0,.14)", animation: "fadeUp .3s ease" }}>
                  <div className="no-print" style={{ padding: "16px 28px", borderBottom: "1px solid #eaeff5", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#64748b", fontWeight: 500 }}>
                      <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="#64748b" strokeWidth="1.5"><path d="M4 1h8l3 3v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1z" /><path d="M10 1v3h3" /></svg>
                      PDF Preview - Portfolio Alignment Report
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button onClick={() => window.print()} className="no-print" style={{ padding: "7px 18px", background: "#0f172a", border: "none", borderRadius: 8, color: "#fff", fontSize: 11, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>Download PDF</button>
                      <button onClick={() => setShowReport(false)} className="no-print" style={{ padding: "7px 14px", background: "#fafbfc", border: "1px solid #e2e8f0", borderRadius: 8, color: "#64748b", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>Close</button>
                    </div>
                  </div>

                  <div style={{ padding: "36px 44px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28, paddingBottom: 20, borderBottom: "2px solid #0f172a" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                        <BGVLogo />
                        <div>
                          <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: ".12em", color: "#a0aec0", textTransform: "uppercase" }}>Benhamou Global Ventures</div>
                          <div style={{ fontFamily: SERIF, fontSize: 21, fontWeight: 400, color: "#0f172a", fontStyle: "italic" }}>Portfolio Alignment Report</div>
                        </div>
                      </div>
                      <div style={{ textAlign: "right", fontSize: 10, color: "#a0aec0" }}>
                        <div>{new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</div>
                        <div style={{ fontWeight: 600, color: "#64748b", letterSpacing: ".06em" }}>CONFIDENTIAL</div>
                      </div>
                    </div>

                    <div style={{ background: "#fafbfc", borderRadius: 10, padding: "16px 20px", marginBottom: 24, border: "1px solid #eaeff5" }}>
                      <div style={{ fontSize: 9, textTransform: "uppercase", letterSpacing: ".1em", color: "#a0aec0", marginBottom: 6 }}>Portfolio Company</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: "#0f172a" }}>{portfolioInput}</div>
                      <div style={{ fontSize: 11.5, color: "#64748b", marginTop: 5, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
                        {portfolioDescription}
                      </div>
                    </div>

                    <div style={{ fontFamily: SERIF, fontSize: 17, fontWeight: 400, fontStyle: "italic", color: "#0f172a", marginBottom: 16 }}>Gap Analysis - Top Matches</div>
                    {matchResults?.map((m, i) => {
                      const sc = getScoreColor(m.gms);
                      return (
                        <div key={i} style={{ padding: "14px 18px", border: "1px solid #eaeff5", borderRadius: 10, marginBottom: 8 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a", flex: 1 }}>{m.gap}</div>
                            <div style={{ fontFamily: SERIF, fontSize: 20, fontWeight: 400, color: sc.accent, marginLeft: 16 }}>{m.gms}</div>
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "#a0aec0", marginBottom: 6 }}>
                            <CompanyLogo name={m.company_need} s={11} />
                            <span>{m.company_need}</span>
                            <span style={{ margin: "0 2px" }}>|</span>
                            <span>Capability: {(m.capability_match * 100).toFixed(0)}%</span>
                            {m.industry_match !== null && (
                              <>
                                <span style={{ margin: "0 2px" }}>|</span>
                                <span>Industry: {(m.industry_match * 100).toFixed(0)}%</span>
                              </>
                            )}
                          </div>
                          <div style={{ fontSize: 10, fontWeight: 700, color: sc.accent, marginTop: 8, marginBottom: 2, textTransform: "uppercase", letterSpacing: ".05em" }}>Strategic Gap Content</div>
                          <div style={{ fontSize: 11, color: "#475569", lineHeight: 1.55 }}>{m.gap_text || m.evidence}</div>
                        </div>
                      );
                    })}

                    <div style={{ marginTop: 20, padding: "14px 18px", background: "#f0f9ff", borderRadius: 10, borderLeft: "3px solid #2563eb" }}>
                      <div style={{ fontSize: 10, fontWeight: 700, color: "#1e40af", marginBottom: 5, textTransform: "uppercase", letterSpacing: ".06em" }}>
                        Strategic Investment Memo
                      </div>

                      {!memoLoading && !memoError && memoItems.length > 1 && (
                        <div style={{ fontSize: 10.5, color: "#64748b", marginBottom: 8, fontStyle: "italic" }}>
                        Based on top {memoItems.length} GMS matches and retrieved evidence chunks.
                        </div>
                      )}

                      {!memoLoading && !memoError && memoItems.length === 1 && memoMeta && (
                        <div style={{ fontSize: 10.5, color: "#64748b", marginBottom: 8, fontStyle: "italic" }}>
                          Based on highest-ranked GMS match:{" "}
                          <span style={{ fontWeight: 600, color: "#334155", fontStyle: "normal" }}>
                            {memoMeta.target_company}
                          </span>
                          {" "}(GMS {Math.round(Number(memoMeta.display_score || 0))}) - {memoMeta.gap_title}
                        </div>
                      )}

                      <div style={{ fontSize: 12, color: "#334155", lineHeight: 1.65, whiteSpace: "pre-wrap" }}>
                        {memoLoading && (
                          <span style={{ color: "#64748b", fontStyle: "italic" }}>
                            Generating investment memo from GMS matches and retrieved evidence… (this can take 3-8 seconds)
                          </span>
                        )}

                        {!memoLoading && memoError && (
                          <span style={{ color: "#b91c1c" }}>Failed to generate memo: {memoError}</span>
                        )}

                        {!memoLoading && !memoError && memo}

                        {!memoLoading && !memoError && !memo && (
                          <span style={{ color: "#64748b", fontStyle: "italic" }}>No memo available.</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>,
              document.body
            )}
          </div>
        )}

        {/* ~~~~~ SETTINGS ~~~~~ */}
        {activeTab === "settings" && (
          <div style={{ padding: "44px 52px", maxWidth: 680, animation: "fadeUp .4s ease" }}>
            <div style={{ marginBottom: 36 }}>
              <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.14em", color: "#a0aec0", textTransform: "uppercase", marginBottom: 8 }}>Configuration</div>
              <h1 style={{ fontFamily: SERIF, fontSize: 32, fontWeight: 400, margin: "0 0 10px", color: "#0f172a", fontStyle: "italic" }}>Settings</h1>
              <p style={{ fontSize: 13.5, color: "#64748b", margin: 0, lineHeight: 1.65 }}>View the ISS event-type weight configuration used for scoring.</p>
            </div>

            {/* ISS Weights */}
            <div style={{ background: "#fafbfc", border: "1px solid #eaeff5", borderRadius: 14, padding: "24px 28px", marginBottom: 16 }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: "#0f172a", marginBottom: 3 }}>ISS Weight Configuration</div>
              <div style={{ fontSize: 11.5, color: "#a0aec0", marginBottom: 20 }}>Event type weights for the Investment Intent Score. Must sum to 100.</div>
              {Object.entries(issWeights).map(([key, val]) => (
                <div key={key} style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
                  <div style={{ width: 175, fontSize: 12.5, color: "#475569", cursor: "help" }} title={
                    key === "Strategic Commitment" ? "Public statements, CEO letters, and explicit declarations of strategic focus." :
                      key === "Resourcing & Talent" ? "Hiring volume, specialized roles, and capital expenditure related to the theme." :
                        key === "Product Shipping" ? "Actual product releases, API endpoints, and feature launches." :
                          key === "Commercial Traction" ? "Reported revenue, user growth, and enterprise adoption metrics." :
                            key === "Partnerships & GTM" ? "Ecosystem plays, alliances, and joint ventures." :
                              key === "Strategic Consistency" ? "Historical alignment and sustained focus over time." : ""
                  }>
                    {key}
                  </div>
                  <div style={{ flex: 1, height: 5, background: "#e8ecf1", borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${val * 2.5}%`, height: "100%", background: "#0f172a", borderRadius: 3, transition: "width .3s" }} />
                  </div>
                  <div style={{ width: 32, fontSize: 14, fontWeight: 500, color: "#0f172a", textAlign: "right", fontFamily: SERIF }}>{val}</div>
                </div>
              ))}
              <div style={{ textAlign: "right", fontSize: 11, color: "#a0aec0", marginTop: 6, paddingTop: 12, borderTop: "1px solid #eaeff5" }}>
                Total: <span style={{ fontWeight: 600, color: "#0f172a" }}>{Object.values(issWeights).reduce((a, b) => a + b, 0)}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
