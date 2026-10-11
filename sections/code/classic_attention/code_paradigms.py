"""
========================================================================================
Modul: Comparație de Cod: 3 Paradigme
Responsabil: Cuclea Luca-Nicolae & Gheorghe Bogdan-Alexandru (Coder)
========================================================================================
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

class PureNumpyMultiHeadAttention:
    def __init__(self, d_model=64, num_heads=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Ponderi inițializate manual (mock weights)
        rng = np.random.RandomState(42)
        self.W_q = rng.randn(d_model, d_model) * 0.02
        self.W_k = rng.randn(d_model, d_model) * 0.02
        self.W_v = rng.randn(d_model, d_model) * 0.02
        self.W_o = rng.randn(d_model, d_model) * 0.02

    def forward(self, x, is_causal=True):
        N, D = x.shape
        # 1. Proiecții liniare: [N, D] @ [D, D] -> [N, D]
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # 2. Reshape & Split în capete: [num_heads, N, d_k]
        Q = Q.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        K = K.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        V = V.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        
        # 3. Scaled Dot-Product Attention
        scores = np.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Mască Cauzală (triunghiulară superioară -> -1e9)
        if is_causal:
            causal_mask = np.tril(np.ones((N, N)))
            scores = np.where(causal_mask == 1, scores, -1e9)
            
        # 5. Softmax numeric stabil manual pe ultima axă
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        if is_causal:
            exp_scores = np.where(causal_mask == 1, exp_scores, 0.0)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 6. Context ponderat: [heads, N, N] @ [heads, N, d_k] -> [heads, N, d_k]
        context = np.matmul(attn_weights, V)
        
        # 7. Concatenare capete înapoi: [N, D]
        context = context.swapaxes(0, 1).reshape(N, D)
        
        # 8. Proiecție de ieșire
        output = np.dot(context, self.W_o)
        return output, attn_weights


class ModernFlashAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal=True):
        B, N, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


def render_code_paradigms():
    st.header("Comparație Sintactică: 3 Moduri de a Implementa Același Mecanism de Atenție")
    st.markdown("""
    Toate cele 3 variante de cod de mai jos realizează **exact același calcul matematic**:
    un modul de **Multi-Head Self-Attention** cu proiecții liniare $Q, K, V$, scalare $\\sqrt{d_k}$, mască cauzală opțională, softmax și proiecție de ieșire $W_O$.
    Observați evoluția sintaxei, a boilerplate-ului și a cantității de cod scris din 2017 până în 2026.
    """)

    code_paradigm = st.radio(
        "Alege paradigma de cod pentru analiză detaliată:",
        [
            "🐍 1. Pure Python & NumPy (De la Zero - Fără biblioteci ML)",
            "🏛️ 2. Google Tensor2Tensor 2017 (Codebase-ul Oficial Vaswani et al.)",
            "⚡ 3. PyTorch 2.x & Hugging Face (Standardul Modern 2026 + FlashAttention)"
        ],
        horizontal=True
    )

    st.divider()

    if code_paradigm.startswith("🐍 1."):
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.subheader("1. Pure Python & NumPy (Matematică Explicită)")
            st.caption("Implementare completă linie-cu-linie fără nicio bibliotecă de deep learning:")
            st.code(r"""import math
import numpy as np

class PureNumpyMultiHeadAttention:
    def __init__(self, d_model=64, num_heads=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Ponderi inițializate manual (mock weights)
        rng = np.random.RandomState(42)
        self.W_q = rng.randn(d_model, d_model) * 0.02
        self.W_k = rng.randn(d_model, d_model) * 0.02
        self.W_v = rng.randn(d_model, d_model) * 0.02
        self.W_o = rng.randn(d_model, d_model) * 0.02

    def forward(self, x, is_causal=True):
        # x shape: [seq_len, d_model]
        N, D = x.shape
        
        # 1. Proiecții liniare: [N, D] @ [D, D] -> [N, D]
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # 2. Reshape & Split în capete: [num_heads, N, d_k]
        Q = Q.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        K = K.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        V = V.reshape(N, self.num_heads, self.d_k).swapaxes(0, 1)
        
        # 3. Scaled Dot-Product Attention
        # MatMul: [heads, N, d_k] @ [heads, d_k, N] -> [heads, N, N]
        scores = np.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Mască Cauzală (triunghiulară superioară -> -1e9)
        if is_causal:
            causal_mask = np.tril(np.ones((N, N)))
            scores = np.where(causal_mask == 1, scores, -1e9)
            
        # 5. Softmax numeric stabil manual pe ultima axă
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        if is_causal:
            exp_scores = np.where(causal_mask == 1, exp_scores, 0.0)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 6. Context ponderat: [heads, N, N] @ [heads, N, d_k] -> [heads, N, d_k]
        context = np.matmul(attn_weights, V)
        
        # 7. Concatenare capete înapoi: [N, D]
        context = context.swapaxes(0, 1).reshape(N, D)
        
        # 8. Proiecție de ieșire
        output = np.dot(context, self.W_o)
        return output, attn_weights
""", language="python")

        with col_c2:
            st.markdown("### 📊 Caracteristici & Analiză")
            st.markdown("""
            - **Linii de cod:** ~45 linii.
            - **Dependențe:** Doar `numpy` și `math`.
            - **Transparență:** 100% didactic — fiecare operație matematică este expusă clar.
            - **Dezavantaje:**
              - Lipsă accelerare automată GPU (fără kernel-uri CUDA optimizate).
              - Softmax-ul necesită scăderea manuală a maximului (`scores - max(scores)`) pentru stabilitate numerică împotriva overflow-ului.
              - Fără autograd (trebuie scris backward pass manual dacă vrem antrenare).
            """)
            
            if st.button("▶️ Rulează Codul NumPy Live", key="run_numpy_mod"):
                x_test = np.random.randn(5, 64)
                module = PureNumpyMultiHeadAttention(d_model=64, num_heads=4)
                out_np, w_np = module.forward(x_test, is_causal=True)
                st.success(f"✓ Execuție reușită! Output shape: {out_np.shape}, Weights shape: {w_np.shape}")
                st.caption(f"Verificare cauzalitate: Pondere către tokeni viitori: {w_np[0, 0, 1]:.1e} (strict 0)")

    elif code_paradigm.startswith("🏛️ 2."):
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.subheader("2. Codebase-ul Oficial din 2017: Tensor2Tensor (TF 1.x)")
            st.caption("Așa arăta codul real scris de Vaswani et al. în repository-ul Google Brain:")
            st.code(r"""# Fragment adaptat din google/tensor2tensor/layers/common_attention.py
import tensorflow as tf
from tensor2tensor.layers import common_attention

def tensor2tensor_self_attention(x, num_heads=4, is_causal=True):
    # x shape: [batch_size, length, d_model]
    d_model = x.get_shape().as_list()[-1]
    length = tf.shape(x)[1]
    
    # 1. În T2T masca cauzală era reprezentată ca un tensor aditiv de bias 4D:
    # Forma [1, 1, length, length] plină cu -1e9 pe triunghiul superior
    if is_causal:
        bias = common_attention.attention_bias_lower_triangle(length)
    else:
        bias = None
        
    # 2. Funcția de bibliotecă din T2T calcula Q, K, V și aplica atenția
    # Notă: Necesita construirea unui graf computațional static (TF 1.x)
    output = common_attention.multihead_attention(
        query_antecedent=x,
        memory_antecedent=None, # None indică Self-Attention
        bias=bias,
        total_key_depth=d_model,
        total_value_depth=d_model,
        output_depth=d_model,
        num_heads=num_heads,
        dropout_rate=0.0,
        attention_type="dot_product",
        name="multihead_attention"
    )
    
    # 3. Execuția necesita o sesiune TF:
    # with tf.Session() as sess:
    #     out_val = sess.run(output, feed_dict={...})
    return output
""", language="python")

        with col_c2:
            st.markdown("### 📊 Caracteristici & Analiză")
            st.markdown("""
            - **Linii de cod:** ~30 linii în script, dar biblioteca din spate (`common_attention.py`) avea **peste 5,000 de linii** de cod intern!
            - **Dependențe:** `tensorflow==1.15.0`, `tensor2tensor`.
            - **Particularități din 2017:**
              - Masca cauzală nu era o mască booleană, ci un **bias aditiv** (`logits += bias`), unde bias-ul avea $-\\infty$ în viitor și $0$ în trecut.
              - Graf static: tensorii nu aveau valori imediate în consolă; trebuiau compilați și rulați într-o sesiune `tf.Session()`.
            - **De ce este istorie azi:**
              - Incompatibil cu hardware-ul modern și bibliotecile Python 3.10+.
              - Abstracția modulară a fost înlocuită de biblioteci axate pe flexibilitate (PyTorch).
            """)

    else:
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            st.subheader("3. PyTorch 2.x & Hugging Face (Standardul Modern 2026)")
            st.caption("Cum se implementează în prezent folosind optimizări la nivel de kernel CUDA (FlashAttention):")
            st.code(r"""import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernFlashAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # O singură proiecție combinată pentru Q, K, V (economie de memorie GPU)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal=True):
        # x shape: [batch, seq_len, d_model]
        B, N, D = x.shape
        
        # 1. Proiecție QKV combinată într-un singur apel: [B, N, 3, heads, head_dim]
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # -> [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. PyTorch 2.0+ Native Scaled Dot-Product Attention
        # Apelează automat FlashAttention-2 sau Memory-Efficient Attention în CUDA!
        # Rezolvă scaling, masking cauzal și softmax într-un singur kernel fuzionat!
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        # 3. Concatenare și proiecție de ieșire
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
""", language="python")

        with col_c2:
            st.markdown("### 📊 Caracteristici & Analiză")
            st.markdown(r"""
            - **Linii de cod:** **16 linii** (reducere masivă de boilerplate!).
            - **Inovație cheie:** `F.scaled_dot_product_attention(q, k, v, is_causal=True)` (introdus în PyTorch 2.0).
            - **Beneficii hardware colosale:**
              - **FlashAttention-2 Fused Kernel:** Nu mai materializează deloc matricea $N \times N$ în memoria globală VRAM, calculând softmax-ul prin *tiling* în SRAM-ul GPU-ului!
              - Complexitatea memoriei scade de la $\mathcal{O}(N^2)$ la $\mathcal{O}(N)$ în faza de calcul!
              - Viteza crește de 2x-4x comparativ cu codul din 2017.
            """)
            
            if st.button("▶️ Rulează Modulul PyTorch Modern Live", key="run_pytorch_mod"):
                x_torch = torch.randn(2, 5, 64)
                mod_torch = ModernFlashAttention(d_model=64, num_heads=4)
                out_torch = mod_torch(x_torch, is_causal=True)
                st.success(f"✓ Execuție PyTorch 2.x reușită! Output tensor shape: {list(out_torch.shape)}")
                st.caption(f"Calculul cauzal s-a executat nativ în interiorul PyTorch fără generare manuală de măști.")

    st.divider()
    st.subheader("Tabel Sinoptic Comparativ între cele 3 Paradigme")
    summary_data = {
        "Criteriu": [
            "Linii de cod scrise",
            "Paradigme de execuție",
            "Mascare cauzală",
            "Softmax",
            "Consum memorie VRAM",
            "Accelerare Hardware",
            "Unde este folosit azi?"
        ],
        "1. Pure Python / NumPy": [
            "~45 linii",
            "Eager (CPU)",
            "Manuală (`np.where` & `np.tril`)",
            "Manual cu `np.exp(s - max(s))`",
            "Mare (alocă matrice N x N)",
            "Fără GPU / SIMD lent",
            "Didactic / Interviuri tehnice"
        ],
        "2. Google Tensor2Tensor (2017)": [
            "~30 linii (+5000 linii lib)",
            "Static Graph (`tf.Session`)",
            "Bias aditiv 4D (`+ bias`)",
            "`tf.nn.softmax` clasic",
            r"Pătratic $\mathcal{O}(N^2)$",
            "NVIDIA CUDA basic",
            "Arhivă istorică (depășit)"
        ],
        "3. PyTorch 2.x Modern (2026)": [
            "~16 linii",
            "Eager / `torch.compile`",
            "Parametru nativ `is_causal=True`",
            "Kernel fuzionat în CUDA",
            r"Liniar $\mathcal{O}(N)$ (FlashAttention)",
            "Tensor Cores / FlashAttn-2",
            "LLaMA, GPT-4, Mistral, Industrie"
        ]
    }
    st.table(pd.DataFrame(summary_data))
