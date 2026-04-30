import streamlit as st
from app import load, Indexer, analyze, calculate
import re

st.set_page_config(page_title="MOURAG", page_icon="📄", layout="wide")

# Initialize
@st.cache_resource
def init():
    mous = load()
    idx = Indexer(mous)
    return mous, idx

mous, idx = init()

# UI
st.title("MOURAG")
st.caption(f"Search across {len(mous)} MOUs")

with st.form("search"):
    c1, c2 = st.columns([5, 1])
    query = c1.text_input("Enter your query", placeholder="e.g., Explain your Project or Startup")
    limit = c2.number_input("Results", 1, 10, 5)
    btn = st.form_submit_button("Search")

if btn and query:
    info = analyze(query)
    ids, dists = idx.search(query)
    
    results = []
    for i, mid in enumerate(ids):
        mou = idx.mous.get(mid)
        if mou:
            score = calculate(mou, info, dists[i])
            results.append((mou, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:limit]
    
    # Show intent
    if info["intent"] != "general":
        st.info(f"Detected intent: **{info['intent']}**")
    if info["departments"]:
        st.success(f"Matched departments: {', '.join(info['departments'][:3])}")
    
    st.divider()
    
    # Results
    for rank, (mou, score) in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.subheader(f"{rank}. {mou['title']}")
                st.write(f"**Partner:** {mou['partner']}")
                
                if mou["departments"]:
                    st.write(f"**Departments:** {', '.join(mou['departments'][:3])}")
                
                if mou["domains"]:
                    st.write(f"**Domains:** {', '.join(mou['domains'][:3])}")
                
                if mou["highlights"]:
                    st.write("**Highlights:**")
                    for h in mou["highlights"][:2]:
                        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', h)[:100]
                        st.write(f"- {clean}")
                
                if mou["start"] or mou["end"]:
                    st.write(f"Valid: {mou['start']} → {mou['end']}")
            
            with col2:
                st.metric("Score", f"{score:.2f}")
            
            st.divider()
