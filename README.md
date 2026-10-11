# 🧠 Transformer & SWA Interactive Lab
> **LLMs & Cognition 2026** • FMI UniBuc  
> Aplicație interactivă Streamlit pentru prezentarea articolelor:
> 1. *"Attention Is All You Need"* (Vaswani et al., 2017)
> 2. *"Sliding-Window Beats Linear Attention"* (Jolicoeur-Martineau et al., 2026)

---

## 🚀 Pornire Rapidă (Quick Start)

### 1. Clonare Repository
```bash
git clone <URL_REPOSITORY_GITHUB>
cd presentation_app
```

### 2. Creare și Activare Mediu Virtual (Recomandat)

- **Pe Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(Dacă PowerShell blochează rularea de scripturi, rulează o singură dată: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **Pe Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalare Dependințe
```bash
pip install -r requirements.txt
```

### 4. Lansare Aplicație
```bash
streamlit run main.py
```
Aplicația se va deschide automat în browser la adresa: **`http://localhost:8501`**.

---

## 📦 Dependințe (`requirements.txt`)

- `streamlit>=1.30.0` (Interfața web interactivă)
- `torch>=2.0.0` (Calcule tensoriale și Shape Inspector)
- `numpy>=1.24.0` (Operații numerice și simulări)
- `pandas>=2.0.0` (Manipulare date și tabele)
- `plotly>=5.18.0` (Grafice interactive)

---

## 👥 Cum este Împărțit Proiectul (Arhitectură & Roluri)

Aplicația este construită pe o **arhitectură complet modulară**.  
Fiecare tab principal are un fișier Python dedicat în folderul `sections/`, astfel încât **fiecare membru al echipei își poate adăuga partea fără a crea conflicte de merge (`merge conflicts`) cu ceilalți**.

| Tab în Aplicație | Fișier Sursă | Responsabili | Rol Asignat |
| :--- | :--- | :--- | :--- |
| 📌 **Problem Framing** | `sections/problem_framing.py` | Hermeneanu Ionuț Silviu | *Problem Framer* |
| 🔬 **Methodology** | `sections/methodology.py` | Leahu Teodor Filip & Szocs Luca | *Methodology Experts* |
| 💻 **Code (Classic Attention)** | `sections/code/classic_attention/` | Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru | *Coders* |
| 🌊 **Code (Other Attention / SWA)**| `sections/code/other_attention/` | Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru | *Coders* |
| 📊 **Results & Analysis** | `sections/results.py` | Gusev Roman | *Results Interpreter & Challenger* |
| 🎛️ **Main Orchestrator** | `main.py` | Luca Cuclea / Întreaga echipă | *Entrypoint & Navigation Layout* |

---

## 🛠️ Cum Contribui (Ghid pentru Membrii Echipei)

Fiecare membru trebuie să modifice **doar fișierul asociat secțiunii sale**:

1. **Pentru Problem Framing (`sections/problem_framing.py`):**
   - Modifică funcția `render_problem_framing()`.
   - Poți adăuga text formatat (`st.markdown`), formule LaTeX (`st.latex`), imagini (`st.image`) sau callout-uri (`st.info`, `st.warning`).

2. **Pentru Methodology (`sections/methodology.py`):**
   - Modifică funcția `render_methodology()`.
   - Adaugă detaliile arhitecturale (Encoder-Decoder, Positional Encoding, Multi-Head Attention equations etc.).

3. **Pentru Results (`sections/results.py`):**
   - Modifică funcția `render_results()`.
   - Adaugă grafice (Plotly / Streamlit native), tabele de scoruri BLEU, comparații de complexitate O(N) vs O(N²), analize critice.

4. **Pentru Code (`sections/code/...`):**
   - Cod modular pentru demonstrații interactive, shape tracking, winograd schema etc.

---

## 🌿 Workflow Recomandat pe Git

Pentru a colabora fără bătăi de cap:

1. **Actualizează codul înainte de a începe să lucrezi:**
   ```bash
   git pull origin main
   ```
2. **(Opțional dar recomandat) Lucrează pe un branch separat:**
   ```bash
   git checkout -b feature/numele-tau
   ```
3. **După ce ai adăugat modificările și ai testat cu `streamlit run main.py`:**
   ```bash
   git add .
   git commit -m "Add: descriere scurta a modificarilor"
   git push origin feature/numele-tau
   ```
4. **Deschide un Pull Request (PR) pe GitHub** sau dă merge în `main` dacă lucrați direct pe main.
