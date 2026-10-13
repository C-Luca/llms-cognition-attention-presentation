# 🧠 Transformer & SWA Interactive Lab
> **LLMs & Cognition 2026** • FMI UniBuc  
> Interactive Streamlit application for presenting the papers:
> 1. *"Attention Is All You Need"* (Vaswani et al., 2017)
> 2. *"Sliding-Window Beats Linear Attention"* (Jolicoeur-Martineau et al., 2026)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/C-Luca/llms-cognition-attention-presentation.git
cd llms-cognition-attention-presentation
```

### 2. Create and Activate a Virtual Environment (Recommended)

- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell restricts running scripts, run once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run main.py
```
The application will automatically open in your browser at: **`http://localhost:8501`**.

---

## 📦 Dependencies (`requirements.txt`)

- `streamlit>=1.30.0` (Interactive web UI)
- `torch>=2.0.0` (Tensor operations & Shape Inspector)
- `numpy>=1.24.0` (Numerical computing & simulations)
- `pandas>=2.0.0` (Data manipulation & tabular views)
- `plotly>=5.18.0` (Interactive visualizations & plots)

---

## 🌿 Recommended Git Workflow

To collaborate smoothly with team members:

1. **Pull the latest changes before starting work:**
   ```bash
   git pull origin main
   ```
2. **(Optional but recommended) Work on a dedicated branch:**
   ```bash
   git checkout -b feature/your-name
   ```
3. **After adding changes and testing locally with `streamlit run main.py`:**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   git push origin feature/your-name
   ```
4. **Open a Pull Request (PR) on GitHub** or merge into `main` if collaborating directly on main.
