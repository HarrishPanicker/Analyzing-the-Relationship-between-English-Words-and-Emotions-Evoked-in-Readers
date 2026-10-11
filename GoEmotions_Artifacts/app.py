"""
GoEmotions Explorer: Ekman vs a data-driven emotion scheme (BERT / ELECTRA / RoBERTa).

Run from the folder containing GoEmotions_Artifacts (or inside it):
    pip install streamlit torch transformers pandas pillow altair
    streamlit run app.py

Optional: drop methodology_flow.png next to app.py (or in the artifacts folder)
and it shows up on the Overview tab.
"""
import html
import string
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="GoEmotions Explorer", page_icon="🧠", layout="wide")

MAX_LEN = 128
MAX_INPUT_WORDS = 9  # predictions degrade on longer text, so inputs are capped
MAX_EXPLAIN_WORDS = 80
ROBERTA_MAX_TOKENS = 512  # RoBERTa's hard input limit (incl. <s> and </s>)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["BERT", "ELECTRA", "RoBERTa"]
SCHEMES = {"Ekman": "ekman", "New scheme": "custom"}
METRICS = ["Precision", "Accuracy", "F1 (Micro)", "F1 (Macro)"]

# Used if the *_results_summary.csv files can't be parsed.
FALLBACK_RESULTS = pd.DataFrame(
    [
        ("Ekman", "BERT", 0.6625, 0.7677, 0.7677, 0.6499),
        ("Ekman", "ELECTRA", 0.6634, 0.7777, 0.7777, 0.6476),
        ("Ekman", "RoBERTa", 0.6645, 0.7782, 0.7782, 0.6557),
        ("New scheme", "BERT", 0.7460, 0.7528, 0.7528, 0.7385),
        ("New scheme", "ELECTRA", 0.7438, 0.7568, 0.7568, 0.7452),
        ("New scheme", "RoBERTa", 0.7514, 0.7627, 0.7627, 0.7535),
    ],
    columns=["Scheme", "Model"] + METRICS,
)

# Edit these. Try a few and keep the ones where the two schemes give different answers.
EXAMPLES = {
    "😒 Sarcasm": "Well, that's just great. Another Monday.",
    "🥹 Proud": "I'm so proud of you, you finally did it!",
    "🤢 Disgust": "This is disgusting, I can't even look at it.",
    "😨 Fear": "I don't know what's next, and it's terrifying.",
    "🙏 Mixed": "I can't believe you did that, thanks so much!",
}


# ---------- file helpers ----------
def default_root() -> str:
    here = Path(__file__).resolve().parent
    if (here / "models").exists():
        return str(here)
    if (here / "GoEmotions_Artifacts").exists():
        return str(here / "GoEmotions_Artifacts")
    return str(here)


@st.cache_data(show_spinner=False)
def locate(root_str: str, name: str) -> str:
    """Find a file by name anywhere under root (or next to app.py). Returns '' if missing."""
    root = Path(root_str)
    here = Path(__file__).resolve().parent
    for base in (root, here):
        if base.exists():
            hit = next(base.rglob(name), None)
            if hit:
                return str(hit)
    return ""


def find_model_dir(base: Path):
    if not base.exists():
        return None
    if (base / "config.json").exists():
        return base
    for p in sorted(base.rglob("config.json")):
        return p.parent
    return None


def parse_labels(data) -> dict:
    if isinstance(data, dict):
        for key in ("id2label", "labels", "classes"):
            if key in data:
                return parse_labels(data[key])
        first = next(iter(data))
        if str(first).isdigit():
            return {int(k): str(v) for k, v in data.items()}
        return {int(v): str(k) for k, v in data.items()}
    return {i: str(v) for i, v in enumerate(data)}


def find_label_file(root: Path, scheme: str):
    name = f"{scheme}_labels.json"
    for p in (root / name, root / "models" / name, root / "models" / scheme / name):
        if p.exists():
            return p
    return next(root.rglob(name), None) if root.exists() else None


def load_label_names(model_dir: Path, config) -> dict:
    id2label = {int(k): v for k, v in config.id2label.items()}
    if all(str(v).startswith("LABEL_") for v in id2label.values()):
        for folder in (model_dir, model_dir.parent):
            for name in ("labels.json", "label_map.json", "id2label.json", "label_names.json"):
                f = folder / name
                if f.exists():
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        return {i: str(v) for i, v in enumerate(data)}
                    if isinstance(data, dict):
                        first_key = next(iter(data))
                        if str(first_key).isdigit():
                            return {int(k): str(v) for k, v in data.items()}
                        return {int(v): str(k) for k, v in data.items()}
    return id2label


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    if df[first].dtype == object or str(first).startswith("Unnamed"):
        df = df.set_index(first)
        df.index.name = None
    return df


def pick(cols, *keys):
    """First column whose lowercase name contains every key."""
    for c in cols:
        lc = str(c).lower().replace("_", " ")
        if all(k in lc for k in keys):
            return c
    return None


def canon_model(name: str) -> str:
    n = str(name).lower()
    for m in ("roberta", "electra", "bert"):  # roberta first: "bert" is inside it
        if m in n:
            return {"roberta": "RoBERTa", "electra": "ELECTRA", "bert": "BERT"}[m]
    return str(name)


# ---------- model + prediction ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_dir_str: str, label_file_str: str = ""):
    model_dir = Path(model_dir_str)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    labels = None
    if label_file_str:
        try:
            labels = parse_labels(json.loads(Path(label_file_str).read_text(encoding="utf-8")))
            if len(labels) != model.config.num_labels:
                labels = None
        except Exception:
            labels = None
    if labels is None:
        labels = load_label_names(model_dir, model.config)
    multi = getattr(model.config, "problem_type", None) == "multi_label_classification"
    return tokenizer, model, labels, multi


def run_probs(texts, bundle) -> np.ndarray:
    """Probabilities for a batch of texts -> array [n_texts, n_labels]."""
    tokenizer, model, _, multi = bundle
    enc = tokenizer(
        list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits) if multi else torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


def predict(text: str, bundle) -> pd.DataFrame:
    labels = bundle[2]
    probs = run_probs([text], bundle)[0]
    df = pd.DataFrame(
        {"emotion": [labels[i] for i in range(len(probs))], "probability": probs, "idx": range(len(probs))}
    )
    return df.sort_values("probability", ascending=False).reset_index(drop=True)


def explain(text: str, bundle, idx: int):
    """Occlusion: drop each word, see how much the target probability falls."""
    all_words = text.split()
    words = all_words[:MAX_EXPLAIN_WORDS]
    if len(words) == 0:
        return None
    if len(words) == 1:  # nothing to compare against: the single word is the whole signal
        return words, np.array([1.0]), []
    # Occlude every copy of a word together, so repeated words ("love love love") still register.
    keys = [w.lower().strip(string.punctuation) or w.lower() for w in all_words]
    variants = [" ".join(all_words)] + [
        " ".join(w for w, k in zip(all_words, keys) if k != keys[i]) for i in range(len(words))
    ]
    p = np.clip(run_probs(variants, bundle)[:, idx], 1e-6, 1 - 1e-6)
    z = np.log(p) - np.log1p(-p)  # log-odds: still sensitive when the model is ~100% sure
    return words, z[0] - z[1:], all_words[MAX_EXPLAIN_WORDS:]


def highlight_html(words, scores, rest=None) -> str:
    scores = np.clip(scores, 0, None)
    top = max(float(scores.max()), 1e-6)
    spans = []
    for w, s in zip(words, scores):
        norm = (float(s) / top) ** 0.5  # sqrt scaling so weaker words stay visible
        style = "padding:2px 5px;border-radius:5px;margin-right:3px;display:inline-block;"
        if norm > 0.2:
            style += f"background:rgba(255,99,71,{0.2 + 0.7 * norm:.2f});"
        spans.append(f'<span style="{style}">{html.escape(w)}</span>')
    for w in rest or []:  # words beyond the cap: shown plain, not analysed
        spans.append(f'<span style="padding:2px 5px;margin-right:3px;display:inline-block;opacity:.6">{html.escape(w)}</span>')
    return f'<div style="line-height:2.3;font-size:1.05rem;text-align:left">{"".join(spans)}</div>'


@st.cache_resource(show_spinner=False)
def get_tokenizer(model_dir_str: str):
    try:
        return AutoTokenizer.from_pretrained(Path(model_dir_str))
    except Exception:
        return AutoTokenizer.from_pretrained("roberta-base")


def count_tokens(model_dir_str: str, text: str) -> int:
    return len(get_tokenizer(model_dir_str)(text, truncation=False)["input_ids"])


def analyse(model_dir, label_file, text):
    if model_dir is None:
        return {"error": "Model folder not found. Check the artifacts path in the sidebar."}
    try:
        bundle = load_model(str(model_dir), str(label_file) if label_file else "")
        return {"bundle": bundle, "df": predict(text, bundle)}
    except Exception as e:
        return {"error": f"Could not run this model: {e}"}


def render_result(title: str, res: dict, text: str, explain_on: bool):
    st.subheader(title)
    if "error" in res:
        st.error(res["error"])
        return
    df, bundle = res["df"], res["bundle"]
    top = df.iloc[0]
    st.metric("Top emotion", str(top["emotion"]).capitalize(), f"{top['probability']:.1%}")
    st.bar_chart(df.set_index("emotion")["probability"])
    if explain_on:
        try:
            out = explain(text, bundle, int(top["idx"]))
            if len(text.split()) < 3:
                st.caption("Very short input: with so few words, highlighting says little (every word carries all the signal).")
            if out:
                st.caption(f"Words pushing towards **{top['emotion']}** (darker = stronger, measured in log-odds)")
                st.markdown(highlight_html(*out), unsafe_allow_html=True)
                if len(text.split()) > MAX_EXPLAIN_WORDS:
                    st.caption(f"Only the first {MAX_EXPLAIN_WORDS} words are analysed; the rest is shown faded.")
        except Exception as e:
            st.caption(f"Word highlighting unavailable: {e}")
    with st.expander("Full scores"):
        st.dataframe(
            df.drop(columns="idx").style.format({"probability": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )


# ---------- results + charts ----------
@st.cache_data(show_spinner=False)
def load_results(root_str: str):
    """Combine ekman/custom summary CSVs. Falls back to hard-coded numbers if columns aren't recognised."""
    try:
        frames = []
        for scheme_name, key in SCHEMES.items():
            f = locate(root_str, f"{key}_results_summary.csv")
            if not f:
                raise FileNotFoundError(key)
            df = pd.read_csv(f)
            cols = list(df.columns)
            model_col = pick(cols, "model") or cols[0]
            mapping = {
                "Precision": pick(cols, "prec"),
                "Accuracy": pick(cols, "acc"),
                "F1 (Micro)": pick(cols, "micro") or pick(cols, "f1", "micro"),
                "F1 (Macro)": pick(cols, "macro", "f1") or pick(cols, "macro"),
            }
            if any(v is None for v in mapping.values()):
                raise ValueError("unrecognised columns")
            out = pd.DataFrame({"Scheme": scheme_name, "Model": df[model_col].map(canon_model)})
            for k, v in mapping.items():
                out[k] = pd.to_numeric(df[v])
            frames.append(out)
        res = pd.concat(frames, ignore_index=True)
        return res, True
    except Exception:
        return FALLBACK_RESULTS.copy(), False


def headline(res: pd.DataFrame):
    try:
        r = res[res["Model"] == "RoBERTa"]
        return r[r["Scheme"] == "Ekman"].iloc[0], r[r["Scheme"] == "New scheme"].iloc[0]
    except Exception:
        return None


def draw_heatmap(df, height=None, annotate=None, xtitle=None, ytitle=None, fmt=None, scheme="blues"):
    df = df.copy()
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    long = df.reset_index(names="row").melt(id_vars="row", var_name="col", value_name="value")
    if fmt is None:
        fmt = ".0f" if (df.round(0) == df).all().all() else ".2f"
    if annotate is None:
        annotate = df.shape[0] <= 14 and df.shape[1] <= 14
    if height is None:
        height = int(min(max(40 * df.shape[0] + 90, 280), 760))
    base = alt.Chart(long).encode(
        x=alt.X("col:N", sort=list(df.columns), title=xtitle, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("row:N", sort=list(df.index), title=ytitle),
    )
    chart = base.mark_rect().encode(
        color=alt.Color("value:Q", scale=alt.Scale(scheme=scheme), legend=None),
        tooltip=["row", "col", alt.Tooltip("value:Q", format=fmt)],
    )
    if annotate:
        cutoff = float(long["value"].max()) * 0.6
        text = base.mark_text(fontSize=13).encode(
            text=alt.Text("value:Q", format=fmt),
            color=alt.condition(alt.datum.value > cutoff, alt.value("white"), alt.value("black")),
        )
        chart = chart + text
    st.altair_chart(chart.properties(height=height), use_container_width=True)


EKMAN_MAP = {  # standard GoEmotions -> Ekman grouping (neutral excluded)
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}


def read_counts(path: str):
    """CSV -> DataFrame[label, count], or None if no numeric column is found."""
    df = pd.read_csv(path)
    obj = [c for c in df.columns if df[c].dtype == object]
    if df.shape[0] == 1 and not obj:  # wide format: labels as columns, one row of counts
        df = df.T.reset_index()
        df.columns = ["label", "count"]
        label_col, value_col = "label", "count"
    else:
        label_col = obj[0] if obj else df.columns[0]
        nums = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
        if not nums:
            return None
        value_col = pick(nums, "count") or pick(nums, "freq") or pick(nums, "total") or nums[0]
    return pd.DataFrame({"label": df[label_col].astype(str), "count": df[value_col]})


def ekman_counts_from(d: pd.DataFrame) -> pd.DataFrame:
    """Sum the 27 emotion counts into Ekman's six (neutral dropped)."""
    lookup = {e: k for k, v in EKMAN_MAP.items() for e in v}
    g = d.assign(group=d["label"].str.strip().str.lower().map(lookup)).dropna(subset=["group"])
    out = g.groupby("group", as_index=False)["count"].sum()
    return out.rename(columns={"group": "label"})


def draw_bar(d: pd.DataFrame, color: str = "#156082"):
    chart = (
        alt.Chart(d)
        .mark_bar(color=color)
        .encode(
            x=alt.X("count:Q", title=None),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=300)),
            tooltip=["label", "count"],
        )
        .properties(height=alt.Step(28))  # fixed height per bar, so no label is ever skipped
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"{len(d)} labels")


def bar_from_csv(path: str, color: str = "#156082"):
    d = read_counts(path)
    if d is None:
        st.dataframe(pd.read_csv(path), use_container_width=True)
        return
    draw_bar(d, color)


def normalize_scheme(data):
    """Turn cluster_scheme.json into {group: [members]}. Handles a few common shapes."""
    if isinstance(data, dict):
        for k in ("clusters", "groups", "mapping", "scheme", "new_scheme"):
            if k in data and isinstance(data[k], (dict, list)):
                return normalize_scheme(data[k])
        if data and all(isinstance(v, (list, tuple)) for v in data.values()):
            return {str(k): [str(x) for x in v] for k, v in data.items()}
        if data and all(isinstance(v, (str, int)) for v in data.values()):
            groups = {}
            for emo, g in data.items():
                groups.setdefault(str(g), []).append(str(emo))
            return groups
    if isinstance(data, list) and data and all(isinstance(v, (list, tuple)) for v in data):
        return {f"Group {i + 1}": [str(x) for x in v] for i, v in enumerate(data)}
    return None


def _cid(k) -> str:
    """Cluster id used for matching: 'Cluster_3' and 3 and '3' all become '3'."""
    k = str(k)
    digits = "".join(ch for ch in k if ch.isdigit())
    return digits if digits else k.lower()


def _find_reps(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if "represent" in str(k).lower():
                return v
        for v in data.values():
            if isinstance(v, dict):
                r = _find_reps(v)
                if r is not None:
                    return r
    return None


def _rep_name(v):
    if isinstance(v, (list, tuple)):
        return _rep_name(v[0]) if v else None
    if isinstance(v, dict):
        for k in ("emotion", "label", "name", "representative"):
            if k in v:
                return _rep_name(v[k])
        return None
    return None if v is None else str(v)


def parse_scheme(data):
    """-> (groups {representative: [members]}, aliases {representative: original cluster key})."""
    reps_raw = _find_reps(data)
    core = data
    if isinstance(data, dict):
        core = {k: v for k, v in data.items() if "represent" not in str(k).lower()}
    groups = normalize_scheme(core)
    if not groups:
        return None, None
    reps = {}
    if isinstance(reps_raw, dict):
        if reps_raw and all(isinstance(v, (int, float)) or str(v).isdigit() for v in reps_raw.values()):
            for emo, cid in reps_raw.items():  # {emotion: cluster id}
                reps[_cid(cid)] = str(emo)
        else:
            for k, v in reps_raw.items():  # {cluster: emotion}
                name = _rep_name(v)
                if name:
                    reps[_cid(k)] = name
    elif isinstance(reps_raw, (list, tuple)):
        for key, v in zip(groups, reps_raw):
            name = _rep_name(v)
            if name:
                reps[_cid(key)] = name
    named, aliases = {}, {}
    for key, members in groups.items():
        rep = reps.get(_cid(key))
        if not rep:
            named[key] = members  # no representative found: keep original name
            continue
        label = rep if rep not in named else f"{rep} ({key})"
        named[label] = members if rep in members else [rep] + members
        aliases[label] = key
    return named, aliases


def show_scheme(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups, _ = parse_scheme(data)
    if not groups:
        st.json(data)
        return
    items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    cols = st.columns(3)
    for i, (g, members) in enumerate(items):
        chips = "".join(
            f'<span style="background:rgba(21,96,130,.18);padding:3px 9px;border-radius:12px;'
            f'margin:0 4px 6px 0;display:inline-block">{html.escape(m)}</span>'
            for m in members
        )
        with cols[i % 3]:
            st.markdown(f"##### {g.capitalize()}")
            st.markdown(chips, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_scheme_groups(root_str: str):
    """({representative: [members]}, {representative: original cluster key}) from cluster_scheme.json."""
    f = locate(root_str, "cluster_scheme.json")
    if not f:
        return {}, {}
    try:
        g, a = parse_scheme(json.loads(Path(f).read_text(encoding="utf-8")))
        return g or {}, a or {}
    except Exception:
        return {}, {}


def compare_schemes(ekman_label: str, new_label: str, groups: dict, aliases: dict = None):
    """Agreement = the new scheme's prediction is the group Ekman's label was folded into.
    Returns (agree: bool, representative of Ekman's group | None)."""
    aliases = aliases or {}
    e, n = ekman_label.lower(), new_label.lower()
    home = None
    for lead, members in groups.items():
        if e == lead.lower() or e in (m.lower() for m in members):
            home = lead
            break
    if home is None:
        return e == n, None
    accepted = {home.lower(), str(aliases.get(home, "")).lower()} | {m.lower() for m in groups[home]}
    return n in accepted, home


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    root_str = st.text_input("Artifacts folder", value=default_root())
    root = Path(root_str)
    st.caption(f"Device: {DEVICE}")
    ekman_dir = find_model_dir(root / "models" / "ekman" / "roberta")
    custom_dir = find_model_dir(root / "models" / "custom" / "roberta")
    ekman_labels = find_label_file(root, "ekman")
    custom_labels = find_label_file(root, "custom")
    st.write("Ekman model:", "✅" if ekman_dir else "❌", "| labels:", "✅" if ekman_labels else "❌")
    st.write("Custom model:", "✅" if custom_dir else "❌", "| labels:", "✅" if custom_labels else "❌")

results, results_from_files = load_results(root_str)

# ---------- main ----------
st.title("GoEmotions Explorer")
st.caption("Does a data-driven emotion scheme beat Ekman's six emotions? BERT vs ELECTRA vs RoBERTa on Reddit comments.")

tab_home, tab_try, tab_results, tab_data = st.tabs(["🏠 Overview", "🧪 Try it", "📊 Results", "🔍 Data & scheme"])

# ===== Overview =====
with tab_home:
    hl = headline(results)
    if hl is not None:
        e, n = hl
        c1, c2, c3 = st.columns(3)
        c1.metric("Macro F1 (RoBERTa)", f"{n['F1 (Macro)']:.4f}", f"{n['F1 (Macro)'] - e['F1 (Macro)']:+.4f} vs Ekman")
        c2.metric("Precision (RoBERTa)", f"{n['Precision']:.4f}", f"{n['Precision'] - e['Precision']:+.4f} vs Ekman")
        c3.metric("Accuracy (RoBERTa)", f"{n['Accuracy']:.4f}", f"{n['Accuracy'] - e['Accuracy']:+.4f} vs Ekman")
        st.markdown(
            "**The trade:** the new scheme gives up a little accuracy but handles rarer emotions far better, "
            "instead of leaning on the common classes."
        )
    st.divider()
    st.subheader("Pipeline")
    flow = locate(root_str, "methodology_flow.png")
    if flow:
        st.image(flow, use_container_width=True)
    else:
        st.markdown(
            "1. Load GoEmotions (58k Reddit comments) → 2. Preprocess → 3. EDA → "
            "4. Build new scheme (TF-IDF + cosine distance) → 5. Train BERT / ELECTRA / RoBERTa → "
            "6. Evaluate → 7. This dashboard"
        )
    st.info("Head to **Try it** to run both schemes on your own text, or **Results** for the full comparison.")

# ===== Try it =====
with tab_try:
    st.session_state.setdefault("input_text", "")

    def set_text(t):
        st.session_state["input_text"] = t

    def clip_words():  # backup: trims to the limit when the box commits
        words = st.session_state.get("input_text", "").split()
        if len(words) > MAX_INPUT_WORDS:
            st.session_state["input_text"] = " ".join(words[:MAX_INPUT_WORDS])
            st.session_state["clipped"] = True

    st.caption("Quick examples")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (label, sample) in zip(ex_cols, EXAMPLES.items()):
        col.button(label, on_click=set_text, args=(sample,), use_container_width=True, key=f"ex_{label}")

    text = st.text_area(
        f"Enter text, max {MAX_INPUT_WORDS} words (press Ctrl+Enter to run)",
        height=110,
        max_chars=150,
        key="input_text",
        on_change=clip_words,
        placeholder="e.g. I can't believe you did that, thanks so much!",
    )
    # Live blocker: stops a 10th word being typed. Relies on Streamlit's page DOM; clip_words() is the fallback.
    components.html(
        """
<script>
const MAX = __MAX__;
const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
function attach() {
  const doc = window.parent.document;
  doc.querySelectorAll("textarea").forEach(t => {
    if (t.dataset.wordLimit) return;
    if (!(t.getAttribute("aria-label") || "").startsWith("Enter text")) return;
    t.dataset.wordLimit = "1";
    t.addEventListener("input", () => {
      const words = t.value.split(/\\s+/).filter(Boolean);
      if (words.length > MAX) {
        setter.call(t, words.slice(0, MAX).join(" "));
        t.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
  });
}
setInterval(attach, 400);
</script>
""".replace("__MAX__", str(MAX_INPUT_WORDS)),
        height=0,
    )
    if st.session_state.pop("clipped", False):
        st.caption(f"Trimmed to {MAX_INPUT_WORDS} words.")
    explain_on = st.checkbox("Highlight the words that drove each prediction", value=True)
    n_words = len(text.split())

    if not text.strip():
        st.caption(f"Pick an example or type something (max {MAX_INPUT_WORDS} words) to see both schemes side by side.")
    elif n_words > MAX_INPUT_WORDS:
        st.error(f"Too long: {n_words} words. The limit is {MAX_INPUT_WORDS}, because predictions get unreliable on longer text. Shorten it to run.")
    else:
        st.caption(f"{n_words} / {MAX_INPUT_WORDS} words")
        res_e = analyse(ekman_dir, ekman_labels, text)
        res_c = analyse(custom_dir, custom_labels, text)
        if "df" in res_e and "df" in res_c:
            a, b = str(res_e["df"].iloc[0]["emotion"]), str(res_c["df"].iloc[0]["emotion"])
            groups, aliases = load_scheme_groups(root_str)
            agree, home = compare_schemes(a, b, groups, aliases)
            folded = f" (folded into **{home.capitalize()}** in the new scheme)" if home and home.lower() != a.lower() else ""
            if agree:
                st.success(f"Schemes agree: Ekman says **{a.capitalize()}**{folded}, new scheme says **{b.capitalize()}**")
            else:
                st.warning(f"Schemes disagree: Ekman says **{a.capitalize()}**{folded}, new scheme says **{b.capitalize()}**")
        col_e, col_c = st.columns(2)
        with col_e:
            render_result("Ekman scheme (RoBERTa)", res_e, text, explain_on)
        with col_c:
            render_result("New scheme (RoBERTa)", res_c, text, explain_on)

# ===== Results =====
with tab_results:
    if not results_from_files:
        st.caption("Showing built-in result numbers (couldn't parse the summary CSVs).")
    t_sum, t_conf, t_label = st.tabs(["Model comparison", "Confusion matrices", "Per-emotion scores"])

    with t_sum:
        hl = headline(results)
        if hl is not None:
            e, n = hl
            st.success(
                f"RoBERTa: Macro F1 **{e['F1 (Macro)']:.4f} → {n['F1 (Macro)']:.4f}** "
                f"({n['F1 (Macro)'] - e['F1 (Macro)']:+.4f}), accuracy "
                f"{e['Accuracy']:.4f} → {n['Accuracy']:.4f} ({n['Accuracy'] - e['Accuracy']:+.4f})."
            )
        metric = st.selectbox("Metric to plot", METRICS, index=METRICS.index("F1 (Macro)"))
        chart = (
            alt.Chart(results)
            .mark_bar()
            .encode(
                x=alt.X("Scheme:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Scheme:N", scale=alt.Scale(range=["#A5B4C0", "#156082"])),
                column=alt.Column("Model:N", sort=MODELS, title=None),
                tooltip=["Scheme", "Model", alt.Tooltip(f"{metric}:Q", format=".4f")],
            )
            .properties(width=150, height=300)
        )
        st.altair_chart(chart)

        def _best(s):
            out = []
            for i, v in s.items():
                grp = results.loc[results["Scheme"] == results.at[i, "Scheme"], s.name]
                out.append("font-weight:700;background-color:rgba(46,160,67,.3)" if v == grp.max() else "")
            return out

        st.dataframe(
            results.style.format({m: "{:.4f}" for m in METRICS}).apply(_best, subset=METRICS),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Green = best model within each scheme.")

    with t_conf:
        c1, c2 = st.columns([1, 1])
        model_name = c1.radio("Model", MODELS, index=2, horizontal=True, key="cm_model")
        norm = c2.checkbox("Normalise rows (show %)", value=True)
        cols = st.columns(2)
        for col, (scheme_name, key) in zip(cols, SCHEMES.items()):
            with col:
                st.subheader(scheme_name)
                f = locate(root_str, f"confusion_{key}_{model_name.lower()}.csv")
                if not f:
                    st.info("Confusion matrix file not found.")
                    continue
                cm = load_csv(f)
                if norm:
                    cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
                draw_heatmap(cm, xtitle="Predicted", ytitle="Actual", annotate=True, height=520)
        st.caption("Rows = actual emotion, columns = predicted. A strong diagonal means fewer mix-ups.")

    with t_label:
        raw = {}
        for scheme_name, key in SCHEMES.items():
            for m in MODELS:
                f = locate(root_str, f"{key}_per_label_{m.lower()}.csv")
                if f:
                    raw[(scheme_name, m)] = load_csv(f)
        if not raw:
            st.info("Per-emotion files not found.")
        else:
            first_cols = list(next(iter(raw.values())).columns)
            options = {"F1": pick(first_cols, "f1"), "Precision": pick(first_cols, "precision"), "Recall": pick(first_cols, "recall")}
            options = {k: v for k, v in options.items() if v}
            if not options:
                st.dataframe(next(iter(raw.values())), use_container_width=True)
            else:
                metric_label = st.radio("Metric", list(options), horizontal=True, key="pl_metric")
                col_name = options[metric_label]
                cols = st.columns(2)
                for col, scheme_name in zip(cols, SCHEMES):
                    with col:
                        st.subheader(scheme_name)
                        parts = {m: raw[(scheme_name, m)][col_name] for m in MODELS if (scheme_name, m) in raw and col_name in raw[(scheme_name, m)].columns}
                        if not parts:
                            st.info("No data for this scheme.")
                            continue
                        table = pd.DataFrame(parts)
                        draw_heatmap(table, fmt=".2f", annotate=True, height=int(min(60 * len(table) + 90, 560)))
                st.caption("Rare emotions usually score lowest. Compare how each scheme handles them.")
                with st.expander("Raw tables"):
                    for (scheme_name, m), df in raw.items():
                        st.markdown(f"**{scheme_name} / {m}**")
                        st.dataframe(df, use_container_width=True)

# ===== Data & scheme =====
with tab_data:
    t_counts, t_scheme, t_corr = st.tabs(["Label counts", "New scheme", "Emotion relationships"])

    with t_counts:
        f_all = locate(root_str, "emotion_counts.csv")
        f_ekman = locate(root_str, "ekman_label_distribution.csv")
        f_new = locate(root_str, "custom_label_distribution.csv")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ekman scheme label distribution**")
            if f_ekman:
                bar_from_csv(f_ekman, "#A5B4C0")
            elif f_all:
                d_all = read_counts(f_all)
                if d_all is None:
                    st.info("Couldn't read emotion_counts.csv")
                else:
                    draw_bar(ekman_counts_from(d_all), "#A5B4C0")
                    st.caption("Summed from the 27 emotion counts using the standard Ekman mapping (neutral excluded).")
            else:
                st.info("emotion_counts.csv not found")
        with c2:
            st.markdown("**New scheme label distribution**")
            bar_from_csv(f_new) if f_new else st.info("custom_label_distribution.csv not found")
        st.divider()
        st.markdown("**Original emotion frequency**")
        bar_from_csv(f_all) if f_all else st.info("emotion_counts.csv not found")

    with t_scheme:
        f = locate(root_str, "cluster_scheme.json")
        if not f:
            st.info("cluster_scheme.json not found")
        else:
            st.caption("Emotions grouped by clustering on TF-IDF cosine distance. Each group is led by its most frequent emotion.")
            show_scheme(f)

    with t_corr:
        which = st.radio(
            "View",
            ["Annotator co-occurrence", "Text similarity (TF-IDF)"],
            horizontal=True,
        )
        fname = "correlation_matrix.csv" if which.startswith("Annotator") else "text_similarity_matrix.csv"
        f = locate(root_str, fname)
        if not f:
            st.info(f"{fname} not found")
        else:
            draw_heatmap(load_csv(f), height=700, annotate=False)
            st.caption(
                "How often annotators tag two emotions together."
                if which.startswith("Annotator")
                else "How similar the vocabulary of two emotions is (cosine similarity). This drives the new grouping."
            )