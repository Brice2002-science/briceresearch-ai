# BRICERESEARCH AI

Assistant de rédaction scientifique (interface chat, fond vert, comptes utilisateurs) fondé sur les
Notes de méthodologie de la recherche scientifique du **Pr A. B. Fandohan** (EForT/UNA).
Moteur : **Groq** (llama-3.3-70b). Comptes + sauvegarde des discussions : **Supabase**.

---

## 0. Révoquer la clé exposée (obligatoire)
La clé Groq que tu as partagée dans le chat est compromise. Va sur **console.groq.com → API Keys**,
supprime-la, crée-en une **neuve**. Elle ira dans un *secret*, jamais dans le code.

## 1. Créer la base Supabase (comptes + discussions)
1. **supabase.com → New project** (gratuit). Note le mot de passe de la base.
2. **Project Settings → API** : copie **Project URL** (→ `SUPABASE_URL`) et la clé **anon public**
   (→ `SUPABASE_ANON_KEY`).
3. **SQL Editor → New query** : colle tout `schema.sql`, puis **Run**.
4. **Authentication → Providers → Email** : pour une inscription fluide entre camarades, **désactive
   "Confirm email"** (sinon chacun doit valider un e-mail avant de se connecter).

## 2. Mettre les fichiers sur GitHub
Crée un dépôt et pousse : `app.py`, `requirements.txt`, `schema.sql`, `logo-icon.svg`,
`logo-icon.png`, et le dossier `.streamlit/` (avec `config.toml`).
Ne mets **jamais** de clé dans ces fichiers.

## 3. Déployer et obtenir l'URL

### Option A — Streamlit Community Cloud (recommandé, gratuit)
1. **share.streamlit.io → New app** → choisis ton dépôt et `app.py`.
2. **Advanced settings → Secrets**, colle (au format TOML) :
   ```toml
   GROQ_API_KEY = "ta_nouvelle_cle_groq"
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_ANON_KEY = "eyJhbGci...la_cle_anon..."
   ```
3. **Deploy**. Tu obtiens une URL type `https://briceresearch.streamlit.app` à envoyer à tes camarades.

### Option B — Hugging Face Spaces (gratuit)
1. **huggingface.co → New Space → SDK : Streamlit**. Uploade les mêmes fichiers.
2. **Settings → Variables and secrets** : ajoute `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`.
3. Le Space se construit et te donne une URL publique.

## 4. Utilisation
Chaque camarade ouvre l'URL, **crée un compte** (e-mail + mot de passe), et discute. Ses conversations
sont sauvegardées et rechargées à sa prochaine connexion. Focus (Résumé / Discussion / Conclusion /
Général) dans la barre latérale.

---

## Sécurité & bonnes pratiques
- Les clés vivent dans les *secrets* de la plateforme (côté serveur), jamais dans le code ni le navigateur.
- Les mots de passe sont gérés et chiffrés par **Supabase Auth** ; l'app ne les voit jamais.
- La **RLS** (Row Level Security) garantit que chacun n'accède qu'à ses propres discussions.
- Quota Groq gratuit : suffisant pour un petit groupe. En cas de forte charge, change de modèle ou passe
  à un plan payant.

## Intégrité académique
C'est une aide à la **rédaction**, pas un générateur de recherche. L'auteur reste responsable :
vérifier chaque affirmation, citer ses propres sources, passer le texte à un anti-plagiat avant soumission.

## Fichiers
- `app.py` — l'application Streamlit.
- `requirements.txt` — dépendances.
- `schema.sql` — tables Supabase + RLS.
- `.streamlit/config.toml` — thème vert.
- `logo-icon.svg` / `logo-icon.png` — logo.
