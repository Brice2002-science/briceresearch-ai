# BRICERESEARCH AI

Assistant de rédaction scientifique (interface chat, fond vert, comptes utilisateurs) fondé sur les
Notes de méthodologie de la recherche scientifique du **Pr A. B. Fandohan** (EForT/UNA).
Moteur : **Groq** (llama-3.3-70b). Comptes + sauvegarde des discussions : **Supabase**.

---

## État du déploiement

| Étape | Statut |
|---|---|
| Dépôt GitHub | ✅ https://github.com/Brice2002-science/briceresearch-ai |
| Base Supabase (tables + RLS) | ✅ projet `lfpzszauclmxahdvbrez` |
| Inscription sans confirmation e-mail | ✅ activée |
| Clé Groq | ⬜ à créer |
| Déploiement Streamlit Cloud | ⬜ à faire |

## 0. Révoquer la clé exposée (obligatoire)
La clé Groq partagée en clair est compromise. Va sur **console.groq.com → API Keys**,
supprime-la, crée-en une **neuve**. Elle ira dans un *secret*, jamais dans le code.

## 1. Base Supabase — ✅ déjà fait
Projet `lfpzszauclmxahdvbrez` (Central EU / Frankfurt). Le `schema.sql` a été exécuté
(2 tables + Row Level Security), et « Confirm email » est désactivé pour que tes camarades
s'inscrivent sans passer par une validation d'e-mail.

Pour refaire cette étape sur un autre projet : **SQL Editor → New query** → coller `schema.sql` → **Run**,
puis **Authentication → Sign In / Providers → Confirm email : off**.

## 2. GitHub — ✅ déjà fait
Le dépôt public contient `app.py`, `requirements.txt`, `schema.sql`, les logos et `.streamlit/config.toml`.
`.gitignore` exclut `.streamlit/secrets.toml` : **aucune clé ne peut partir sur GitHub**.

## 3. Déployer et obtenir l'URL

### Option A — Streamlit Community Cloud (recommandé, gratuit)
1. **share.streamlit.io → New app → From existing repo**
   - Repository : `Brice2002-science/briceresearch-ai`
   - Branch : `main` · Main file path : `app.py`
2. **Advanced settings → Secrets**, colle le contenu de ton `.streamlit/secrets.toml` local
   (format TOML, les 3 clés `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`).
3. **Deploy**. Tu obtiens une URL type `https://briceresearch-ai.streamlit.app` à envoyer à tes camarades.

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
