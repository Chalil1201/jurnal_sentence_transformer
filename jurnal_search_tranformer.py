import os
import json
import urllib.request
import urllib.parse
import re
import math
import random
from datetime import datetime
import time

# Environment Config
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
SAVE_DIR = "D:/jurnal_search_history"
os.makedirs(SAVE_DIR, exist_ok=True)
API_KEY = "API_KEY"

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# --- UTILS ---
def translate_term(term):
    term = term.strip()
    if not term or len(term) < 2: return [term]
    
    # Protected technical terms
    protected = ["ft-transformer", "tabtransformer", "tabnet", "tf-transformer", "transformer", 
                 "large language model", "cnn", "yolo", "random forest", "svm", "xgboost"]
    if term.lower() in protected:
        return [term]
        
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(term)}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            res = json.loads(r.read().decode('utf-8'))
            translated = "".join([s[0] for s in res[0]]).strip()
            if translated.lower() != term.lower():
                return [term, translated]
    except:
        pass
    return [term]

def decompose_and_expand(query_input):
    q_low = query_input.lower()
    
    # 1. Capture Methods (Priority)
    method_patterns = {
        "FT-Transformer": r"ft-transformer[s]?",
        "TabTransformer": r"tabtransformer[s]?",
        "TabNet": r"tabnet[s]?",
        "TF-Transformer": r"tf-transformer[s]?",
        "Transformer": r"\btransformer[s]?\b",
        "Large Language Model": r"large language model[s]?|llm[s]?",
        "CNN": r"\bcnn\b|convolutional neural network[s]?",
        "YOLO": r"\byolo\b",
        "Deep Learning": r"deep learning",
        "Machine Learning": r"machine learning"
    }
    
    found_methods = []
    q_remainder = query_input
    for name, pat in method_patterns.items():
        if re.search(pat, q_low):
            found_methods.append(name)
            q_remainder = re.sub(pat, "", q_remainder, flags=re.IGNORECASE)

    # 2. Cleanup Remainder
    stop_words = ["menggunakan", "dan", "dengan", "pada", "untuk", "using", "and", "with", "on", "for", "in", "of", "the", "a", "an", "yang", "terhadap", "dalam"]
    clean_text = re.sub(r'[,.\-_/]+', ' ', q_remainder)
    words = [w.strip() for w in clean_text.split() if w.strip().lower() not in stop_words and len(w.strip()) > 1]
    
    # 3. Classify Domains & Tasks
    domain_keywords = ["warga", "binaan", "narapidana", "lapas", "prisoner", "inmate", "prison", "padi", "daun", "rice", "leaf", "penyakit", "disease", "literatur", "akademik", "academic", "literature", "karyawan", "rekrutmen", "produk", "ulasan", "ecommerce", "kinerja", "employee", "recruitment", "product", "review", "performance"]
    task_keywords = ["prediksi", "risiko", "resiko", "pelanggaran", "disiplin", "klasifikasi", "prediction", "risk", "misconduct", "violation", "classification", "detection", "deteksi", "rekomendasi", "sentimen", "analisis", "recommendation", "sentiment", "analysis"]
    
    found_domains = []
    found_tasks = []
    
    # Multi-word checks
    words_joined = " ".join(words).lower()
    if "warga binaan" in words_joined or ("warga" in words_joined and "binaan" in words_joined):
        found_domains.append("warga binaan")
    if "pelanggaran disiplin" in words_joined:
        found_tasks.append("pelanggaran disiplin")
    if "prediksi risiko" in words_joined or "prediksi resiko" in words_joined:
        found_tasks.append("prediksi risiko")
    if "rekrutmen karyawan" in words_joined or "rekrutmen" in words_joined:
        found_domains.append("rekrutmen karyawan")
    if "ulasan produk" in words_joined or "produk e-commerce" in words_joined:
        found_domains.append("ulasan produk")
    if "kinerja karyawan" in words_joined or "kinerja" in words_joined:
        found_domains.append("kinerja karyawan")

    for w in words:
        wl = w.lower()
        if wl in task_keywords:
            if wl not in str(found_tasks).lower(): found_tasks.append(wl)
        elif wl in domain_keywords:
            if wl not in str(found_domains).lower(): found_domains.append(wl)
        elif len(wl) > 3:
            if wl not in str(found_domains).lower(): found_domains.append(wl)

    # 4. Global Expansion (Translation)
    analysis = {"domain": [], "task": [], "methods": []}
    
    for d in found_domains:
        analysis["domain"].extend(translate_term(d))
        if d.lower() in ["warga binaan", "narapidana"]:
            analysis["domain"].extend(["prisoner", "inmate", "offender"])
        elif d.lower() in ["rekrutmen karyawan", "karyawan"]:
            analysis["domain"].extend(["employee recruitment", "hiring", "job applicant", "talent acquisition"])
        elif d.lower() in ["ulasan produk", "produk", "ecommerce"]:
            analysis["domain"].extend(["product review", "e-commerce feedback", "customer review"])
        elif d.lower() in ["kinerja karyawan", "kinerja"]:
            analysis["domain"].extend(["employee performance", "job performance", "work productivity"])
            
    for t in found_tasks:
        analysis["task"].extend(translate_term(t))
        if t.lower() in ["pelanggaran disiplin"]:
            analysis["task"].extend(["misconduct", "disciplinary violation"])
        elif t.lower() in ["rekomendasi"]:
            analysis["task"].extend(["recommender system", "recommendation system"])
        elif t.lower() in ["sentimen", "analisis"]:
            analysis["task"].extend(["sentiment analysis", "opinion mining"])
        elif t.lower() in ["prediksi"]:
            analysis["task"].extend(["prediction model", "predictive analytics"])
            
    for m in found_methods:
        analysis["methods"].extend(translate_term(m))

    # Deduplicate
    for k in analysis:
        analysis[k] = list(dict.fromkeys(analysis[k]))

    # 5. Build Unified Query
    unified_parts = []
    for k in ["domain", "task", "methods"]:
        if analysis[k]:
            terms = ['"' + x + '"' if ' ' in x else x for x in analysis[k]]
            unified_parts.append("(" + " OR ".join(terms) + ")")
    
    unified_query = " AND ".join(unified_parts)
    return analysis, unified_query if unified_query else query_input

# --- STREAMLIT UI ---
st.set_page_config(page_title="Jurnal Search Sentence-Transformers", layout="wide")
st.title("📚 Jurnal Search Sentence-Transformers")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi")
use_scholar = st.sidebar.checkbox("Google Scholar (SerpApi)", value=False)
use_openalex = st.sidebar.checkbox("OpenAlex (Gratis)", value=True)
use_semantic = st.sidebar.checkbox("Semantic Scholar (Gratis)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("📁 Riwayat Pencarian")
history_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".json")], key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)
selected_history = st.sidebar.selectbox("Pilih Riwayat:", ["Pencarian Baru"] + history_files)

if "current_results" not in st.session_state: st.session_state.current_results = None
if "current_page" not in st.session_state: st.session_state.current_page = 0
if "last_history" not in st.session_state: st.session_state.last_history = "Pencarian Baru"

if selected_history != st.session_state.last_history:
    st.session_state.current_page = 0
    st.session_state.last_history = selected_history
    st.session_state.current_results = None

query_input = st.text_input("Masukkan judul/topik penelitian:", value="" if selected_history == "Pencarian Baru" else selected_history.rsplit('_', 2)[0].replace('_', ' '))

if st.button("Search Jurnal") and selected_history == "Pencarian Baru":
    if not (use_scholar or use_openalex or use_semantic) or not query_input:
        st.error("⚠️ Masukkan judul dan pilih minimal satu sumber!")
    else:
        all_papers = []
        seen_titles = set()
        
        # 1. Analysis
        analysis, unified_query = decompose_and_expand(query_input)
        st.subheader("🔍 Hasil Analisis Query (Decomposition & Expansion):")
        st.json(analysis)
        st.info(f"**Unified Query (Indo + Global):** {unified_query}")
        
        # 2. Search
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # We run the unified query as the primary search string
        search_targets = [unified_query]
        # Also run the original query to be safe
        if query_input not in search_targets: search_targets.append(query_input)
        
        current_year = datetime.now().year
        start_year = current_year - 5
        
        total_steps = len(search_targets) * sum([use_scholar, use_openalex, use_semantic])
        step = 0
        
        for q_str in search_targets:
            enc_q = urllib.parse.quote(q_str)
            
            # Google Scholar
            if use_scholar:
                step += 1
                status_text.text(f"[Scholar] Searching: {q_str[:40]}...")
                try:
                    url = f"https://serpapi.com/search.json?engine=google_scholar&q={enc_q}&num=100&as_ylo={start_year}&api_key={API_KEY}"
                    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=15) as r:
                        data = json.loads(r.read().decode('utf-8'))
                    for p in data.get('organic_results', []):
                        t = p.get('title', '').strip()
                        if t and t.lower() not in seen_titles:
                            seen_titles.add(t.lower())
                            year_m = re.search(r'\b(20\d{2})\b', p.get('publication_info', {}).get('summary', ''))
                            y = int(year_m.group(1)) if year_m else 0
                            if y >= start_year:
                                pdf = next((res.get('link') for res in p.get('resources', []) if res.get('file_format') == 'PDF'), None)
                                all_papers.append({'title': t, 'authors': p.get('publication_info', {}).get('summary', ''), 'year': y, 'abstract': p.get('snippet', ''), 'link': p.get('link', ''), 'pdf_url': pdf, 'source': 'Google Scholar'})
                except: pass
                progress_bar.progress(step / total_steps)

            # OpenAlex
            if use_openalex:
                step += 1
                status_text.text(f"[OpenAlex] Searching: {q_str[:40]}...")
                try:
                    limit = 25 if use_scholar else 50
                    url = f"https://api.openalex.org/works?search={enc_q}&per_page={limit}&filter=from_publication_date:{start_year}-01-01"
                    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=15) as r:
                        data = json.loads(r.read().decode('utf-8'))
                    for p in data.get('results', []):
                        t = p.get('title')
                        if t and t.lower() not in seen_titles:
                            seen_titles.add(t.lower())
                            all_papers.append({'title': t, 'authors': ", ".join([a.get('author', {}).get('display_name', '') for a in p.get('authorships', [])]), 'year': p.get('publication_year', 0), 'abstract': t, 'link': p.get('doi', ''), 'pdf_url': p.get('open_access', {}).get('oa_url'), 'source': 'OpenAlex'})
                except: pass
                progress_bar.progress(step / total_steps)

            # Semantic Scholar
            if use_semantic:
                step += 1
                status_text.text(f"[Semantic] Searching: {q_str[:40]}...")
                try:
                    time.sleep(1.0)
                    limit = 25 if use_scholar else 50
                    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={enc_q}&limit={limit}&year={start_year}-{current_year}&fields=title,authors,year,abstract,openAccessPdf,url"
                    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=15) as r:
                        data = json.loads(r.read().decode('utf-8'))
                    for p in data.get('data', []):
                        t = p.get('title', '').strip()
                        if t and t.lower() not in seen_titles:
                            seen_titles.add(t.lower())
                            pdf_d = p.get('openAccessPdf')
                            all_papers.append({'title': t, 'authors': ", ".join([a.get('name', '') for a in p.get('authors', [])]), 'year': p.get('year') or 0, 'abstract': p.get('abstract') or '', 'link': p.get('url', ''), 'pdf_url': pdf_d.get('url') if pdf_d else None, 'source': 'Semantic Scholar'})
                except: pass
                progress_bar.progress(step / total_steps)

        # 3. Rank
        if all_papers:
            query_emb = model.encode(query_input, convert_to_tensor=True)
            paper_titles = [p['title'] for p in all_papers]
            paper_embs = model.encode(paper_titles, convert_to_tensor=True)
            sim_scores = util.cos_sim(query_emb, paper_embs)[0]
            
            for i, p in enumerate(all_papers):
                p['relevance_score'] = float(sim_scores[i])
            
            all_papers.sort(key=lambda x: (x['relevance_score'], x['year']), reverse=True)
            
            # Save
            safe_q = re.sub(r'[^a-zA-Z0-9]', '_', query_input)[:50].strip('_')
            filename = f"{safe_q}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(os.path.join(SAVE_DIR, filename), 'w', encoding='utf-8') as f:
                json.dump(all_papers, f, indent=4, ensure_ascii=False)
            
            st.session_state.current_results = all_papers
            st.session_state.current_page = 0
            st.success(f"Berhasil ditemukan {len(all_papers)} records jurnal.")
        else:
            st.error("Hasil pencarian kosong.")
        
        progress_bar.empty()
        status_text.empty()

# --- DISPLAY ---
display_data = None
if selected_history != "Pencarian Baru":
    history_path = os.path.join(SAVE_DIR, selected_history)
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                display_data = json.load(f)
        except: pass
else:
    display_data = st.session_state.current_results

if display_data:
    st.write(f"📊 **Total Records Jurnal**: {len(display_data)} data.")
    
    page_size = 10
    total_pages = math.ceil(len(display_data) / page_size)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ Prev") and st.session_state.current_page > 0:
            st.session_state.current_page -= 1
            st.rerun()
    with c2:
        st.markdown(f"<p style='text-align: center;'>Halaman <b>{st.session_state.current_page + 1}</b> dari <b>{total_pages}</b></p>", unsafe_allow_html=True)
    with c3:
        if st.button("Next ➡️") and st.session_state.current_page < total_pages - 1:
            st.session_state.current_page += 1
            st.rerun()

    start = st.session_state.current_page * page_size
    for idx, p in enumerate(display_data[start:start+page_size]):
        if not isinstance(p, dict): continue
        actual_idx = start + idx + 1
        score_pct = p.get('relevance_score', 0) * 100
        year_str = f"({p.get('year')})" if p.get('year') else ""
        source = p.get('source', 'API')
        
        with st.expander(f"{actual_idx}. [{source}] {p.get('title')} {year_str} - Relevance: {score_pct:.1f}%"):
            st.write(f"**Authors:** {p.get('authors', '-')}")
            st.write(f"**Abstract:** {p.get('abstract', 'N/A')}")
            
            link_col, pdf_col = st.columns([1, 4])
            with link_col:
                if p.get('link'): st.markdown(f"[🔗 Lihat Sumber]({p['link']})")
            with pdf_col:
                if p.get('pdf_url'): st.markdown(f"**[📥 Download PDF]({p['pdf_url']})**")
                else: st.markdown("*PDF tidak terdeteksi.*")
else:
    if selected_history == "Pencarian Baru":
        st.info("Masukkan judul/topik dan klik Search.")
