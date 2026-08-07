"""
BANOUDO AI — assistant de rédaction scientifique (interface chat)
D'après les Notes de méthodologie de la recherche scientifique du Pr A. B. Fandohan (EForT/UNA).

Fonctionnalités
  • Interface conversationnelle (style ChatGPT/Claude), fond vert, logo intégré.
  • Comptes utilisateurs : inscription e-mail + mot de passe (gérés par Supabase Auth).
  • Sauvegarde des discussions par utilisateur (base Supabase, protégée par RLS).
  • Moteur : Groq (llama-3.3-70b) avec le prompt de méthodologie Fandohan.

Aucune clé n'est écrite dans ce fichier. Tout vient des *secrets* de la plateforme :
  GROQ_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
Voir README.md pour le déploiement (Streamlit Cloud / Hugging Face Spaces).
"""

import os
from pathlib import Path

import streamlit as st
from groq import Groq
from supabase import create_client, Client

MODEL = "llama-3.3-70b-versatile"
APP_DIR = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Secrets                                                                      #
# --------------------------------------------------------------------------- #
def secret(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

GROQ_API_KEY = secret("GROQ_API_KEY")
SUPABASE_URL = secret("SUPABASE_URL")
SUPABASE_ANON_KEY = secret("SUPABASE_ANON_KEY")

# --------------------------------------------------------------------------- #
# Méthodologie (Pr Fandohan) — prompt système                                  #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """
Tu es BANOUDO AI, assistant de rédaction scientifique pour chercheurs francophones
(foresterie, écologie, télédétection, environnement). Tu appliques la méthodologie de rédaction
du Pr Adandé Belarmain Fandohan (École de Foresterie Tropicale, UNA, Bénin).

Principes : rigueur (aucune affirmation sans appui dans ce que fournit l'auteur ; n'invente jamais de
résultats, de chiffres ni de références) ; intégrité (tu structures et formules le travail de l'auteur,
tu ne fabriques pas une recherche ; paraphrase toujours — plagiat = 4 mots consécutifs identiques à une
source) ; registre français académique de niveau revue Q1 (ou anglais scientifique soigné).

Quand on te demande une section, applique ces règles.
RÉSUMÉ : 200–300 mots ; aucune abréviation non conventionnelle ; aucune citation ni référence ; temps =
passé composé/présent (FR) ou prétérit (EN) ; il énonce successivement (1) la question centrale, (2) le
gap de connaissance, (3) les objectifs/hypothèses/questions, (4) l'approche méthodologique, (5) les
résultats majeurs, (6) les implications théoriques ET pratiques, (7) la contribution + les perspectives ;
compréhensible sans lire l'article ; sans tableau ni figure.
DISCUSSION : synthétise les résultats majeurs ; compare aux travaux d'autres auteurs (convergences /
divergences) et explique l'originalité ; explique les résultats (mécanismes / driving forces) et les
relie aux théories ; dégage les implications théoriques et pratiques ; tire la contribution clé ; montre
que les résultats répondent aux objectifs de l'introduction ; signale les LIMITES ; propose des
recherches ultérieures. À éviter : sortir du périmètre des objectifs, introduire de nouvelles
données/méthodes, citer des auteurs de façon inexacte, attaquer le travail d'autrui.
CONCLUSION : remets l'étude en contexte ; réponds aux objectifs ; décris l'avancée ; suggère applications
et axes de recherche future liés aux objectifs. À éviter : références, incertitudes (« pourrait être »),
répétition brute des résultats, propos bavard. La conclusion n'est pas un résumé.

Si un élément manque dans ce que fournit l'auteur, dis-le explicitement et demande-le — ne comble jamais
un vide par une invention. Sois structuré, précis et actionnable.
""".strip()

FOCUS_HINT = {
    "Général": "",
    "Résumé": "L'auteur travaille sur son RÉSUMÉ / ABSTRACT.",
    "Discussion": "L'auteur travaille sur sa DISCUSSION.",
    "Conclusion": "L'auteur travaille sur sa CONCLUSION.",
}

# --------------------------------------------------------------------------- #
# Clients                                                                      #
# --------------------------------------------------------------------------- #
@st.cache_resource
def groq_client(api_key: str) -> Groq:
    """La clé fait partie de la signature : sans elle, le client resterait mémorisé
    avec l'ancienne valeur et toute mise à jour du secret serait sans effet."""
    return Groq(api_key=api_key)

def sb_client() -> Client:
    """Client Supabase, avec le jeton de l'utilisateur connecté (pour la RLS)."""
    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    tok = st.session_state.get("access_token")
    if tok:
        try:
            sb.postgrest.auth(tok)
        except Exception:
            pass
    return sb

# --------------------------------------------------------------------------- #
# Base de données : conversations & messages                                   #
# --------------------------------------------------------------------------- #
def db_list_conversations(uid: str):
    try:
        r = (sb_client().table("conversations").select("id,title,created_at")
             .eq("user_id", uid).order("created_at", desc=True).execute())
        return r.data or []
    except Exception:
        return []

def db_create_conversation(uid: str, title: str) -> str | None:
    try:
        r = sb_client().table("conversations").insert(
            {"user_id": uid, "title": title[:80]}).execute()
        return r.data[0]["id"]
    except Exception as e:
        st.error(f"Impossible de créer la discussion : {e}")
        return None

def db_load_messages(conv_id: str):
    try:
        r = (sb_client().table("messages").select("id,role,content,created_at")
             .eq("conversation_id", conv_id).order("created_at").execute())
        return [{"id": m["id"], "role": m["role"], "content": m["content"]}
                for m in (r.data or [])]
    except Exception:
        return []

def db_save_message(conv_id: str, uid: str, role: str, content: str) -> str | None:
    """Enregistre un message et renvoie son id (nécessaire pour « Réessayer »)."""
    try:
        r = sb_client().table("messages").insert(
            {"conversation_id": conv_id, "user_id": uid,
             "role": role, "content": content}).execute()
        return r.data[0]["id"]
    except Exception as e:
        st.warning(f"Message non sauvegardé : {e}")
        return None

def db_delete_conversation(conv_id: str) -> bool:
    """Supprime une discussion. Les messages suivent : la clé étrangère de
    `messages.conversation_id` est déclarée `on delete cascade` dans schema.sql."""
    try:
        sb_client().table("conversations").delete().eq("id", conv_id).execute()
        return True
    except Exception as e:
        st.error(f"Suppression impossible : {e}")
        return False

def db_delete_message(msg_id: str):
    """Supprime un message : sinon une réponse ratée resterait dans l'historique."""
    try:
        sb_client().table("messages").delete().eq("id", msg_id).execute()
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# Authentification                                                             #
# --------------------------------------------------------------------------- #
def set_session(resp):
    if resp and getattr(resp, "session", None) and getattr(resp, "user", None):
        st.session_state.access_token = resp.session.access_token
        st.session_state.user_id = resp.user.id
        st.session_state.user_email = resp.user.email
        return True
    return False

def do_login(email, password):
    try:
        resp = sb_client().auth.sign_in_with_password({"email": email, "password": password})
        if set_session(resp):
            st.rerun()
        else:
            st.error("Connexion impossible. Vérifie l'e-mail / mot de passe (ou confirme ton e-mail).")
    except Exception as e:
        st.error(f"Échec de connexion : {e}")

def do_signup(email, password):
    try:
        resp = sb_client().auth.sign_up({"email": email, "password": password})
        if set_session(resp):
            st.success("Compte créé, te voilà connecté.")
            st.rerun()
        else:
            st.info("Compte créé. Si la confirmation par e-mail est activée, valide-la puis connecte-toi.")
    except Exception as e:
        st.error(f"Échec de l'inscription : {e}")

def do_logout():
    for k in ["access_token", "user_id", "user_email", "current_conv", "messages"]:
        st.session_state.pop(k, None)
    st.rerun()

# --------------------------------------------------------------------------- #
# Branding / CSS                                                               #
# --------------------------------------------------------------------------- #
LOGO_SVG = (APP_DIR / "logo-icon.svg").read_text(encoding="utf-8") if (APP_DIR / "logo-icon.svg").exists() else ""

def inject_css():
    st.markdown("""
    <style>
      :root {
        --ink:#0E1A17;          /* texte principal */
        --muted:#6B837C;        /* texte secondaire */
        --green:#0F6B57;        /* accent principal */
        --green-soft:#E8F2EE;   /* fonds teintés */
        --gold:#B98A34;         /* accent chaud, très parcimonieux */
        --line:rgba(14,26,23,.09);
        --line-strong:rgba(14,26,23,.16);
        --shadow-sm:0 1px 2px rgba(14,26,23,.05);
        --shadow-md:0 2px 4px rgba(14,26,23,.04), 0 12px 28px -18px rgba(14,26,23,.30);
        --shadow-lg:0 2px 6px rgba(14,26,23,.05), 0 32px 64px -36px rgba(14,26,23,.35);
        --r-sm:10px; --r-md:14px; --r-lg:20px;
      }

      /* Barre d'outils Streamlit masquée. L'en-tête reste : il porte le bouton
         d'ouverture de la barre latérale sur mobile. */
      [data-testid="stToolbar"], [data-testid="stToolbarActions"],
      [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
      [data-testid="stDecoration"], [data-testid="manage-app-button"],
      #MainMenu, footer { display:none !important; }
      header[data-testid="stHeader"] { background:transparent; height:2.4rem; }
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapsedControl"] { display:flex !important; }

      /* Halo vert diffus en haut de page : de la profondeur sans salir le blanc. */
      .stApp {
        background:
          radial-gradient(60rem 26rem at 50% -14rem, rgba(15,107,87,.10), transparent 70%),
          #FFFFFF; }

      .block-container { max-width:46rem; padding-top:2.6rem; padding-bottom:8rem; }

      body, .stApp { -webkit-font-smoothing:antialiased; color:var(--ink); }

      /* ---------- Barre latérale ---------- */
      section[data-testid="stSidebar"] {
        background:#FBFDFC; border-right:1px solid var(--line); }
      section[data-testid="stSidebar"] .block-container { padding-top:1.2rem; }
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p strong {
        font-size:11px; letter-spacing:.09em; text-transform:uppercase;
        color:var(--muted); font-weight:600; }
      section[data-testid="stSidebar"] .stButton>button:not([kind="primary"]) {
        background:transparent; border:1px solid transparent; color:var(--ink);
        text-align:left; justify-content:flex-start; font-weight:450;
        font-size:13.5px; padding:.34rem .6rem; }
      section[data-testid="stSidebar"] .stButton>button:not([kind="primary"]):hover {
        background:var(--green-soft); border-color:transparent; color:var(--green); }
      section[data-testid="stSidebar"] .stButton>button[kind="primary"] {
        margin-bottom:.7rem; padding:.5rem; }
      /* Corbeille : invisible au repos, rouge au survol. */
      section[data-testid="stSidebar"] [data-testid="stPopover"] button {
        background:transparent; border:none; color:var(--muted);
        opacity:0; transition:opacity .15s, color .15s; padding:.3rem; min-height:0; }
      section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover
        [data-testid="stPopover"] button { opacity:.65; }
      section[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
        opacity:1; color:#B4413A; }

      /* ---------- Identité ---------- */
      .bra-brand { display:flex; align-items:center; gap:10px; margin:0 0 18px; }
      .bra-brand svg { width:32px; height:32px; border-radius:9px; }
      .bra-brand h2 { font-size:15px; margin:0; color:var(--ink);
        letter-spacing:.01em; font-weight:650; }

      .bra-hero { text-align:center; margin:1.5vh 0 14px; }
      .bra-hero svg { width:58px; height:58px; border-radius:17px;
        box-shadow:0 14px 34px -16px rgba(15,107,87,.6); }
      .bra-hero h1 {
        font-family:'Iowan Old Style',Georgia,'Times New Roman',serif;
        font-size:31px; font-weight:600; letter-spacing:-.015em;
        margin:18px 0 0; color:var(--ink); }

      /* ---------- Messages ---------- */
      .stChatMessage { background:transparent; padding:.15rem 0; }
      [data-testid="stChatMessageContent"] {
        font-size:15.5px; line-height:1.72; color:var(--ink); }
      [data-testid="stChatMessageContent"] h1,
      [data-testid="stChatMessageContent"] h2,
      [data-testid="stChatMessageContent"] h3 {
        font-family:'Iowan Old Style',Georgia,serif; letter-spacing:-.01em; }
      /* Réponse : filet vert à gauche, comme une citation d'appareil critique. */
      .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
        background:linear-gradient(180deg, var(--green-soft), #F4FAF7);
        border:1px solid var(--line); border-left:2.5px solid var(--green);
        border-radius:6px var(--r-md) var(--r-md) 6px;
        padding:.8rem 1.25rem; margin:.5rem 0; }
      /* Message de l'auteur : carte blanche surélevée. */
      .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
        background:#FFFFFF; border:1px solid var(--line);
        border-radius:var(--r-md); padding:.8rem 1.25rem; margin:.5rem 0;
        box-shadow:var(--shadow-sm); }

      /* ---------- Saisie ---------- */
      [data-testid="stChatInput"] {
        border:1px solid var(--line-strong); border-radius:var(--r-md);
        background:#FFFFFF; box-shadow:var(--shadow-md); }
      [data-testid="stChatInput"]:focus-within {
        border-color:var(--green);
        box-shadow:var(--shadow-md), 0 0 0 3px rgba(15,107,87,.10); }

      /* ---------- Boutons ---------- */
      .stButton>button { border-radius:var(--r-sm); font-weight:500; }
      .stButton>button[kind="primary"] { border:none;
        box-shadow:0 1px 2px rgba(15,107,87,.22), 0 10px 22px -14px rgba(15,107,87,.75); }
      .stButton>button[kind="primary"]:hover { filter:brightness(1.07); }

      /* ---------- Cartes ---------- */
      [data-testid="stVerticalBlockBorderWrapper"]:has(.stTabs) {
        background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r-lg);
        padding:8px 24px 12px; box-shadow:var(--shadow-lg); }
      .stTabs [data-baseweb="tab-list"] { gap:20px; }
      .stTabs [data-baseweb="tab-highlight"] { background:var(--green); }
      .stTabs [data-baseweb="tab"] { font-size:14px; }

      /* Pièces jointes : repli discret, pas un bloc lourd. */
      [data-testid="stExpander"] details {
        border:1px solid var(--line); border-radius:var(--r-md);
        background:#FFFFFF; box-shadow:var(--shadow-sm); }
      [data-testid="stExpander"] summary { font-size:13.5px; color:var(--muted); }
      [data-testid="stExpander"] summary:hover { color:var(--green); }

      /* ---------- Actions sous chaque message ---------- */
      .bra-actions { margin-top:.1rem; }
      .stChatMessage [data-testid="stPopover"] button,
      .stChatMessage .stButton>button {
        background:transparent; border:1px solid transparent; color:var(--muted);
        font-size:12px; font-weight:450; padding:.1rem .45rem; min-height:0;
        border-radius:8px; opacity:.6; transition:opacity .15s; }
      .stChatMessage:hover [data-testid="stPopover"] button,
      .stChatMessage:hover .stButton>button { opacity:.85; }
      .stChatMessage [data-testid="stPopover"] button:hover,
      .stChatMessage .stButton>button:hover {
        background:#FFFFFF; border-color:var(--line);
        color:var(--green); opacity:1; }

      .stCaption, [data-testid="stCaptionContainer"] {
        color:var(--muted); font-size:12.5px; }
      hr, [data-testid="stDivider"] { border-color:var(--line); }
      ::selection { background:rgba(15,107,87,.16); }
    </style>
    """, unsafe_allow_html=True)

def brand_block(hero=False):
    cls = "bra-hero" if hero else "bra-brand"
    if hero:
        st.markdown(f'<div class="{cls}">{LOGO_SVG}<h1>BANOUDO AI</h1></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="{cls}">{LOGO_SVG}<h2>BANOUDO AI</h2></div>',
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #
def login_page():
    # Colonne centrale étroite : la carte ne doit pas s'étaler sur toute la largeur.
    _, mid, _ = st.columns([1, 5, 1])
    with mid:
        brand_block(hero=True)
        with st.container(border=True):
            tab_in, tab_up = st.tabs(["Se connecter", "Créer un compte"])
            with tab_in:
                e = st.text_input("E-mail", key="in_e")
                p = st.text_input("Mot de passe", type="password", key="in_p")
                if st.button("Se connecter", type="primary", use_container_width=True):
                    do_login(e, p)
            with tab_up:
                e2 = st.text_input("E-mail", key="up_e")
                p2 = st.text_input("Mot de passe (≥ 6 caractères)", type="password", key="up_p")
                if st.button("Créer mon compte", type="primary", use_container_width=True):
                    if not e2 or len(p2) < 6:
                        st.warning("E-mail requis et mot de passe d'au moins 6 caractères.")
                    else:
                        do_signup(e2, p2)
        st.caption("Tes identifiants sont gérés et chiffrés par Supabase Auth. "
                   "BANOUDO AI ne voit jamais ton mot de passe.")

# --------------------------------------------------------------------------- #
# Documents joints                                                             #
# --------------------------------------------------------------------------- #
# Marqueur qui sépare le message de l'auteur du contenu des fichiers, pour
# pouvoir replier les documents à l'affichage sans les cacher au modèle.
DOC_MARK = "\n\n----- DOCUMENTS JOINTS -----\n"
MAX_CHARS_FILE = 6000      # ~1 500 tokens par fichier
MAX_CHARS_TOTAL = 12000    # ~3 000 tokens au total : le quota Groq est étroit

def extract_text(f) -> str:
    """Texte brut d'un fichier téléversé. Les erreurs sont signalées, pas masquées."""
    name = (f.name or "").lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
        if name.endswith(".docx"):
            import docx
            return "\n".join(p.text for p in docx.Document(f).paragraphs)
        return f.getvalue().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[Lecture impossible : {e}]"

def build_attachments(files) -> str:
    """Assemble les fichiers en un bloc de contexte, tronqué pour tenir le quota."""
    if not files:
        return ""
    parts, budget = [], MAX_CHARS_TOTAL
    for f in files:
        if budget <= 0:
            parts.append(f"\n### {f.name}\n[Non transmis : limite de taille atteinte.]")
            continue
        txt = (extract_text(f) or "").strip()
        cap = min(MAX_CHARS_FILE, budget)
        if len(txt) > cap:
            txt = txt[:cap] + "\n[…texte tronqué pour rester dans le quota…]"
        budget -= len(txt)
        parts.append(f"\n### {f.name}\n{txt or '[Fichier vide ou illisible.]'}")
    return DOC_MARK + "\n".join(parts)

# --------------------------------------------------------------------------- #
# Génération                                                                   #
# --------------------------------------------------------------------------- #
def key_fingerprint() -> str:
    """Décrit la clé lue par l'app sans jamais la divulguer : de quoi distinguer
    « texte d'exemple », « clé tronquée » et « vraie clé rejetée par Groq »."""
    k = GROQ_API_KEY or ""
    return (f"\n\n*(diagnostic : clé lue = {len(k)} caractères, "
            f"préfixe `gsk_` {'présent' if k.startswith('gsk_') else 'ABSENT'})*")

def friendly_error(e: Exception) -> str:
    """Traduit les pannes courantes en langage compréhensible par un étudiant."""
    s = str(e)
    if "invalid_api_key" in s or "401" in s:
        return ("la clé Groq de l'application est invalide ou a été révoquée. "
                "Signale-le à la personne qui administre l'application — "
                "elle doit la mettre à jour dans les secrets Streamlit."
                + key_fingerprint())
    if "rate_limit" in s or "429" in s:
        return ("le quota de l'application est atteint pour le moment. "
                "Réessaie dans quelques minutes.")
    if "model_not_found" in s or "404" in s:
        return f"le modèle « {MODEL} » n'est pas disponible sur ce compte Groq."
    return s

def stream_reply():
    """Flux de la réponse du modèle pour l'historique courant."""
    sys = SYSTEM_PROMPT + ("\n\n" + FOCUS_HINT.get(st.session_state.get("focus", "Général"), ""))
    api_msgs = [{"role": "system", "content": sys}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    stream = groq_client(GROQ_API_KEY).chat.completions.create(
        model=MODEL, messages=api_msgs, temperature=0.4, stream=True)
    for chunk in stream:
        d = chunk.choices[0].delta.content
        if d:
            yield d

def chat_page():
    uid = st.session_state.user_id

    # ---- Panneau latéral : nouvelle discussion + historique ----
    with st.sidebar:
        brand_block()

        if st.button("＋  Nouvelle discussion", type="primary", use_container_width=True):
            st.session_state.current_conv = None
            st.session_state.messages = []
            st.rerun()

        focus = st.selectbox("Focus", list(FOCUS_HINT.keys()))
        st.session_state.focus = focus

        st.markdown("**Mes discussions**")
        convs = db_list_conversations(uid)
        if not convs:
            st.caption("Tes discussions apparaîtront ici dès ton premier message.")
        for c in convs:
            label = (c["title"] or "Sans titre").strip()
            if len(label) > 26:
                label = label[:26].rstrip() + "…"
            active = c["id"] == st.session_state.get("current_conv")
            row, act = st.columns([5, 1], vertical_alignment="center")
            with row:
                if st.button(("●  " if active else "○  ") + label,
                             key="c_" + c["id"], use_container_width=True):
                    st.session_state.current_conv = c["id"]
                    st.session_state.messages = db_load_messages(c["id"])
                    st.rerun()
            with act:
                # Confirmation en deux temps : une suppression ne se rattrape pas.
                with st.popover("🗑", use_container_width=True):
                    st.caption(f"Supprimer « {label} » et tous ses messages ?")
                    if st.button("Oui, supprimer définitivement",
                                 key="del_" + c["id"], type="primary",
                                 use_container_width=True):
                        if db_delete_conversation(c["id"]):
                            if active:
                                st.session_state.current_conv = None
                                st.session_state.messages = []
                            st.rerun()

        st.divider()
        st.caption(st.session_state.user_email)
        if st.button("Se déconnecter", use_container_width=True):
            do_logout()

    # ---- Main chat ----
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("current_conv", None)

    if not st.session_state.messages:
        brand_block(hero=True)
        st.caption("Colle ton titre, tes objectifs, ta méthodo et tes résultats — ou joins tes "
                   "documents avec 📎 — puis demande un résumé, une discussion ou une conclusion. "
                   "Choisis un *Focus* dans la barre latérale.")

    # ---- Messages, avec leurs actions (copier / réessayer) ----
    last = len(st.session_state.messages) - 1
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"], avatar=("🌿" if m["role"] == "assistant" else None)):
            if DOC_MARK in m["content"]:
                head, docs = m["content"].split(DOC_MARK, 1)
                st.markdown(head)
                with st.expander("📎 Documents joints"):
                    st.text(docs)
            else:
                st.markdown(m["content"])

            act = st.container()
            act.markdown('<div class="bra-actions"></div>', unsafe_allow_html=True)
            c1, c2, _ = act.columns([1.1, 1.2, 5])
            with c1:
                with st.popover("📋 Copier", use_container_width=True):
                    st.caption("Utilise l'icône de copie en haut à droite du bloc.")
                    st.code(m["content"], language=None, wrap_lines=True)
            # Réessayer : uniquement sur la dernière réponse de l'assistant
            if m["role"] == "assistant" and i == last:
                with c2:
                    if st.button("↻ Réessayer", key=f"retry_{i}", use_container_width=True):
                        dropped = st.session_state.messages.pop()
                        if dropped.get("id"):
                            db_delete_message(dropped["id"])
                        st.session_state.pending = True
                        st.rerun()

    # ---- Erreur de génération : affichée hors historique, jamais sauvegardée ----
    if st.session_state.get("gen_error"):
        st.error(st.session_state.gen_error)
        if st.button("↻ Réessayer", key="retry_after_error"):
            st.session_state.gen_error = None
            st.session_state.pending = True
            st.rerun()

    # ---- Génération en attente (nouveau message ou réessai) ----
    if st.session_state.get("pending"):
        st.session_state.pending = False
        st.session_state.gen_error = None
        conv = st.session_state.current_conv
        with st.chat_message("assistant", avatar="🌿"):
            try:
                reply = st.write_stream(stream_reply())
            except Exception as e:
                st.session_state.gen_error = "La génération a échoué : " + friendly_error(e)
                reply = None
        if reply:
            mid = db_save_message(conv, uid, "assistant", reply) if conv else None
            st.session_state.messages.append(
                {"id": mid, "role": "assistant", "content": reply})
        st.rerun()

    # ---- Pièces jointes ----
    round_key = st.session_state.get("up_round", 0)
    with st.expander("📎  Joindre des documents (PDF, Word, texte, CSV)"):
        files = st.file_uploader(
            "Le contenu sera transmis avec ton prochain message.",
            accept_multiple_files=True, key=f"uploader_{round_key}",
            type=["pdf", "docx", "txt", "md", "csv"])
        st.caption(f"Extraction limitée à {MAX_CHARS_FILE:,} caractères par fichier "
                   f"et {MAX_CHARS_TOTAL:,} au total, pour rester dans le quota Groq."
                   .replace(",", " "))

    prompt = st.chat_input("Écris ton message…")
    if prompt:
        # créer la conversation au 1er message (titre = texte seul, sans les fichiers)
        if not st.session_state.current_conv:
            st.session_state.current_conv = db_create_conversation(uid, prompt)
        conv = st.session_state.current_conv

        content = prompt + build_attachments(files)
        mid = db_save_message(conv, uid, "user", content) if conv else None
        st.session_state.messages.append({"id": mid, "role": "user", "content": content})
        st.session_state.up_round = round_key + 1     # vide le téléverseur
        st.session_state.pending = True
        st.rerun()

# --------------------------------------------------------------------------- #
# Entrée                                                                       #
# --------------------------------------------------------------------------- #
def main():
    icon = str(APP_DIR / "logo-icon.png") if (APP_DIR / "logo-icon.png").exists() else "🌿"
    st.set_page_config(page_title="BANOUDO AI", page_icon=icon, layout="centered",
                       initial_sidebar_state="expanded")
    inject_css()

    missing = [n for n, v in [("GROQ_API_KEY", GROQ_API_KEY),
                              ("SUPABASE_URL", SUPABASE_URL),
                              ("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)] if not v]
    if missing:
        brand_block(hero=True)
        st.error("Configuration incomplète. Ajoute ces *secrets* sur la plateforme d'hébergement : "
                 + ", ".join("`" + m + "`" for m in missing) + ". Voir README.md.")
        st.stop()

    if st.session_state.get("user_id"):
        chat_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
