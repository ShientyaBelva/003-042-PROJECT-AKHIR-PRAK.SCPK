import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= KONFIGURASI HALAMAN =================
st.set_page_config(
    page_title="SmartWatchRank",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS KUSTOM =================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5e6d3 0%, #e8d5b7 100%);
    }
    [data-testid="stSidebar"] {
        background-color: #3e2723;
    }
    [data-testid="stSidebar"] * {
        color: #f5e6d3 !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        color: black !important;
        background-color: white !important;
        border-radius: 5px;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] input::placeholder {
        color: #4e342e !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] input {
        color: #4e342e !important;
    }
    /* Warna teks pada tag yang sudah dipilih */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {
        color: #4e342e !important;
    }
    /* Warna teks pada opsi yang diketik (saat menulis) */
    [data-testid="stSidebar"] .stMultiSelect input {
        color: #4e342e !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span[data-testid="stMarkdownContainer"] {
        color: #c0c0c0 !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
        background-color: #f5f5f5 !important;
        border-radius: 5px;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] div {
        color: black !important;
    }
    [data-testid="stSidebar"] .stMultiSelect svg {
        fill: #c49a6c !important;
    }
    div[data-testid="stSidebar"] ul[role="listbox"] li {
        color: black !important;
        background-color: white !important;
    }
    .stButton > button {
        background-color: #8b5a2b !important;
        color: white !important;
        border-radius: 8px !important;
        width: 100%;
    }
    h1, h2, h3 {
        color: #4e342e !important;
    }
    .metric-card {
        background-color: #fff8f0;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        border-bottom: 3px solid #c49a6c;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #6d4c41;
    }
    .metric-label {
        font-size: 14px;
        color: #8d6e63;
    }
    .history-card {
        background-color: #fff8f0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #8b5a2b;
    }
    .stTabs {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        flex-wrap: wrap; 
        gap: 8px;
        background-color: transparent;
        border-bottom: none;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #fff8f0;
        border-radius: 30px !important;  
        padding: 8px 24px !important;
        font-size: 1rem;
        font-weight: 500;
        color: #4e342e !important;
        border: 1px solid #c49a6c;
        transition: all 0.2s ease;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8d5b7;
        border-color: #8b5a2b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8b5a2b !important;
        color: white !important;
        border-color: #8b5a2b;
    }
    /* Hilangkan garis bawah default Streamlit */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: stretch;
    }
    .metric-card {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.1s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-value {
        font-size: 32px;  /* sedikit lebih besar */
    }
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #5d3a1a;
    }
</style>
""", unsafe_allow_html=True)

# ================= INISIALISASI SESSION STATE =================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'saved_results' not in st.session_state:
    st.session_state.saved_results = {}
if 'saw_result' not in st.session_state:
    st.session_state.saw_result = None
if 'norm_df' not in st.session_state:
    st.session_state.norm_df = None
if 'norm_details' not in st.session_state:
    st.session_state.norm_details = None

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('smart_watch.csv')
        cols_needed = ['Brand', 'Model', 'Display Size (inches)', 'Resolution', 
                       'Water Resistance (meters)', 'Battery Life (days)', 'Price (USD)']
        available = [c for c in cols_needed if c in df.columns]
        if len(available) < 5:
            st.error("Kolom tidak lengkap. Pastikan file CSV memiliki: Brand, Model, Display Size (inches), Water Resistance (meters), Battery Life (days), Price (USD)")
            return pd.DataFrame()
        df = df[available].copy()
        df['Price (USD)'] = df['Price (USD)'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
        for col in ['Display Size (inches)', 'Water Resistance (meters)', 'Battery Life (days)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# ================= FUNGSI SAW =================
def process_saw(df, weights):
    result = df.copy()
    norm_df = pd.DataFrame(index=df.index)
    norm_details = {}
    
    max_disp = df['Display Size (inches)'].max()
    norm_df['Display Size'] = df['Display Size (inches)'] / max_disp
    norm_details['Display Size'] = f"Benefit, max = {max_disp:.2f}"
    
    max_water = df['Water Resistance (meters)'].max()
    norm_df['Water Resistance'] = df['Water Resistance (meters)'] / max_water
    norm_details['Water Resistance'] = f"Benefit, max = {max_water:.0f}"
    
    max_bat = df['Battery Life (days)'].max()
    norm_df['Battery Life'] = df['Battery Life (days)'] / max_bat
    norm_details['Battery Life'] = f"Benefit, max = {max_bat:.1f}"
    
    min_price = df['Price (USD)'].min()
    norm_df['Price'] = min_price / df['Price (USD)']
    norm_details['Price'] = f"Cost, min = ${min_price:.0f}"
    
    result['SAW_Score'] = (norm_df['Display Size'] * weights['Display Size'] +
                           norm_df['Water Resistance'] * weights['Water Resistance'] +
                           norm_df['Battery Life'] * weights['Battery Life'] +
                           norm_df['Price'] * weights['Price'])
    result['Rank'] = result['SAW_Score'].rank(ascending=False, method='min').astype(int)
    result = result.sort_values('SAW_Score', ascending=False).reset_index(drop=True)
    return result, norm_df, norm_details

# FUNGSI VISUALISASI 
def plot_pie_gradient(data):
    if data.empty:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.text(0.5,0.5,'Data kosong', ha='center',va='center')
        return fig
    brand_counts = data['Brand'].value_counts().head(5)
    cmap = plt.cm.get_cmap('YlOrBr')
    colors = [cmap(i / len(brand_counts)) for i in range(len(brand_counts))]
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.pie(brand_counts.values, labels=brand_counts.index, 
           autopct='%1.1f%%', colors=colors,
           textprops={'fontsize': 8, 'color': '#4e342e'},
           wedgeprops={'edgecolor': 'white', 'linewidth': 0.8})
    for autotext in ax.texts:
        if autotext.get_text().endswith('%'):
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(7)
    n_brands = len(brand_counts)
    ax.set_title(f'Distribusi {n_brands} Brand Terbanyak', fontsize=10, color='#4e342e', fontweight='bold')
    fig.patch.set_facecolor('#f5e6d3')
    return fig

def plot_bar_gradient(data):
    if data.empty:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.text(0.5,0.5,'Data kosong', ha='center',va='center')
        return fig
    top10 = data.head(10).copy()
    top10['Label'] = top10['Brand'] + " " + top10['Model']
    scores = top10['SAW_Score']
    cmap = plt.cm.get_cmap('YlOrBr')
    colors = [cmap(i / len(scores)) for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.barh(top10['Label'], scores, color=colors, edgecolor='#5d3a1a', linewidth=0.8)
    ax.set_xlabel('SAW Score', fontsize=8, color='#4e342e')
    ax.set_title('Top 10 Smartwatch Berdasarkan SAW Score', fontsize=10, color='#4e342e', fontweight='bold')
    ax.invert_yaxis()
    ax.set_facecolor('#faf0e6')
    fig.patch.set_facecolor('#f5e6d3')
    for bar, val in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left', fontsize=7, color='#4e342e')
    plt.tight_layout()
    return fig

def plot_line_chart_harga_saw(data):
    if data.empty or len(data) < 2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.text(0.5,0.5,'Data tidak cukup untuk plot', ha='center',va='center')
        fig.patch.set_facecolor('#f5e6d3')
        return fig
    sorted_data = data.sort_values('Price (USD)')
    x = sorted_data['Price (USD)']
    y = sorted_data['SAW_Score']
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(x, y, color='#8b5a2b', linewidth=1.5, marker='o', markersize=3,
            markerfacecolor='#c49a6c', markeredgecolor='#5d3a1a', linestyle='-', label='SAW Score')
    ax.fill_between(x, y, alpha=0.2, color='#c49a6c')
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min if y_max != y_min else 0.1
    y_padding = y_range * 0.1
    ax.set_ylim(bottom=y_min - y_padding, top=y_max + y_padding)
    ax.set_xlabel('Harga (USD)', fontsize=8, color='#4e342e')
    ax.set_ylabel('SAW Score', fontsize=8, color='#4e342e')
    ax.set_title('Tren SAW Score terhadap Harga', fontsize=10, color='#4e342e', fontweight='bold')
    ax.set_facecolor('#faf0e6')
    fig.patch.set_facecolor('#f5e6d3')
    ax.grid(True, linestyle='--', alpha=0.3, color='#8d6e63')
    ax.legend(loc='best', fontsize=7)
    plt.tight_layout()
    return fig

# ================= MAIN =================
def main():
    st.title("⌚ SmartWatchRank")
    st.markdown("### PEMILIHAN SMARTWATCH TERBAIK")
    st.markdown("#### Menggunakan Metode Simple Additive Weighting (SAW) - SCPK 2025/2026")
    st.markdown("---")
    
    df = load_data()
    if df.empty:
        st.stop()
    
    with st.sidebar:
        st.image("smartwatch.png", width=250)
        st.markdown("## ⚙️ Bobot Kriteria")
        st.markdown("Total bobot akan otomatis = 1.00 (bobot relatif)")
        
        raw_display = st.slider("📱 Ukuran Layar (Benefit)", 0.0, 1.0, 0.25, 0.01)
        raw_water = st.slider("💧 Ketahanan Air (Benefit)", 0.0, 1.0, 0.25, 0.01)
        raw_battery = st.slider("🔋 Baterai (Benefit)", 0.0, 1.0, 0.25, 0.01)
        raw_price = st.slider("💰 Harga (Cost)", 0.0, 1.0, 0.25, 0.01)
        
        total_raw = raw_display + raw_water + raw_battery + raw_price
        if total_raw == 0:
            total_raw = 1
        weights = {
            'Display Size': raw_display / total_raw,
            'Water Resistance': raw_water / total_raw,
            'Battery Life': raw_battery / total_raw,
            'Price': raw_price / total_raw
        }
        
        st.markdown(f"**Total Bobot (setelah normalisasi):** 1.00")
        st.markdown("**Bobot akhir yang digunakan:**")
        st.markdown(f"- 📱 Ukuran Layar: {weights['Display Size']:.3f}")
        st.markdown(f"- 💧 Ketahanan Air: {weights['Water Resistance']:.3f}")
        st.markdown(f"- 🔋 Baterai: {weights['Battery Life']:.3f}")
        st.markdown(f"- 💰 Harga: {weights['Price']:.3f}")
        
        st.markdown('<hr style="margin: 1rem 0; border: 1px solid #f5e6d3;">', unsafe_allow_html=True)
        st.metric("📊 Jumlah Smartwatch", len(df))
        st.metric("🏷️ Jumlah Brand", df['Brand'].nunique())
        brand_filter = st.multiselect(
            "🔍 Filter Brand",
            options=sorted(df['Brand'].unique()),
            default=[],
            placeholder="Pilih brand..."
        )
        
        if st.button("🔢 Hitung SAW", use_container_width=True, type="primary"):
            # Ambil bobot dan filter brand dari state saat ini (sudah terdefinisi)
            with st.spinner("Menghitung SAW..."):
                result_full, norm, details = process_saw(df, weights)
                if brand_filter:
                    result_filtered = result_full[result_full['Brand'].isin(brand_filter)]
                else:
                    result_filtered = result_full
                st.session_state.saw_result = result_filtered
                st.session_state.norm_df = norm
                st.session_state.norm_details = details
                st.success("Perhitungan selesai!")
        st.markdown('<hr style="margin: 1rem 0; border: 1px solid #f5e6d3;">', unsafe_allow_html=True)
        
        calculation_name = st.text_input(
            "📝 Nama Perhitungan",
            value=f"Perhitungan {datetime.now().strftime('%H:%M:%S')}",
            key="calc_name_input"
        )
        
        if st.button("💾 Simpan Perhitungan ke Riwayat", use_container_width=True):
            if st.session_state.saw_result is None or st.session_state.saw_result.empty:
                st.warning("Tidak ada hasil perhitungan. Klik 'Hitung SAW' terlebih dahulu.")
            else:
                nama_untuk_disimpan = calculation_name.strip()
                if nama_untuk_disimpan == "":
                    nama_untuk_disimpan = f"Perhitungan {datetime.now().strftime('%H:%M:%S')}"
                result = st.session_state.saw_result  # ambil dari session state
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry = {
                    'id': len(st.session_state.history) + 1,
                    'timestamp': timestamp,
                    'name': nama_untuk_disimpan,
                    'weights': weights.copy(),   # bobot saat itu (bisa disimpan)
                    'top_watch': {
                        'brand': result.iloc[0]['Brand'],
                        'model': result.iloc[0]['Model'],
                        'score': float(result.iloc[0]['SAW_Score'])
                    },
                    'total_watches': len(result)
                }
                st.session_state.history.insert(0, entry)
                st.session_state.saved_results[entry['id']] = result.copy()
                st.success(f"✅ '{nama_untuk_disimpan}' tersimpan!")
                st.rerun()                
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📱 Data Smartwatch", "🏆 Hasil SAW", "📜 Riwayat Perhitungan", "📊 Visualisasi", "ℹ️ Tentang"])
    
    with tab1:
        st.markdown("## 📱 Data Smartwatch")
        st.markdown("Berikut adalah data lengkap smartwatch yang tersedia dalam sistem.")
        
        # Tampilkan dataframe tanpa filter (seluruh data)
        st.dataframe(df, use_container_width=True, height=500)
        
        # Opsi download sebagai CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data sebagai CSV",
            data=csv,
            file_name="smartwatch_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # HASIL SAW
    with tab2:
        result_filtered = st.session_state.saw_result
        norm_df = st.session_state.norm_df
        norm_details = st.session_state.norm_details
        
        if result_filtered is not None and not result_filtered.empty:
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            with col_met1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(result_filtered)}</div><div class="metric-label">Total Smartwatch</div></div>', unsafe_allow_html=True)
            with col_met2:
                avg_price = result_filtered['Price (USD)'].mean()
                st.markdown(f'<div class="metric-card"><div class="metric-value">${avg_price:.0f}</div><div class="metric-label">Harga Rata-rata</div></div>', unsafe_allow_html=True)
            with col_met3:
                avg_bat = result_filtered['Battery Life (days)'].mean()
                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_bat:.1f} d</div><div class="metric-label">Baterai Rata-rata</div></div>', unsafe_allow_html=True)
            with col_met4:
                avg_water = result_filtered['Water Resistance (meters)'].mean()
                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_water:.0f} m</div><div class="metric-label">Ketahanan Air</div></div>', unsafe_allow_html=True)
            winner = result_filtered.iloc[0]
            
            st.markdown("<br>", unsafe_allow_html=True) 
            col_win1, col_win2 = st.columns([1,2])
            with col_win1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #c49a6c, #8b5a2b); border-radius: 20px; padding: 20px; text-align: center; color: white;">
                    <h1 style="color: white;">🥇</h1>
                    <h2 style="color: white;">#1 Pilihan Terbaik</h2>
                    <p>SAW Score: {winner['SAW_Score']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_win2:
                st.markdown(f"""
                <div style="background-color: #fff8f0; border-radius: 15px; padding: 20px;">
                    <h2>🏆 {winner['Brand']} {winner['Model']}</h2>
                    <table style="width:100%">
                        <tr><td>📱 Ukuran Layar</td><td><b>{winner['Display Size (inches)']}"</b></td></tr>
                        <tr><td>💧 Ketahanan Air</td><td><b>{winner['Water Resistance (meters)']}m</b></td></tr>
                        <tr><td>🔋 Baterai</td><td><b>{winner['Battery Life (days)']} days</b></td></tr>
                        <tr><td>💰 Harga</td><td><b>${winner['Price (USD)']}</b></td></tr>
                        <tr><td>🎯 SAW Score</td><td><b>{winner['SAW_Score']:.4f}</b></td></tr>
                        <tr><td>📊 Rank</td><td><b>#{winner['Rank']}</b></td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True) 
            st.markdown("### 📋 Hasil Perhitungan SAW Lengkap")
            display_cols = ['Rank', 'Brand', 'Model', 'Display Size (inches)', 'Water Resistance (meters)', 
                            'Battery Life (days)', 'Price (USD)', 'SAW_Score']
            display_df = result_filtered[display_cols].copy()
            display_df['SAW_Score'] = display_df['SAW_Score'].round(4)
            st.dataframe(display_df, use_container_width=True, height=400)
            with st.expander("📐 Detail Normalisasi SAW"):
                st.markdown("**Parameter Normalisasi:**")
                for k, v in norm_details.items():
                    st.markdown(f"- **{k}**: {v}")
                st.markdown("**Nilai Normalisasi (10 baris pertama):**")
                st.dataframe(norm_df.head(10), use_container_width=True)
        else:
            st.info("🔍 Belum ada hasil. Klik tombol 'Hitung SAW' di sidebar terlebih dahulu.")
    
    # RIWAYAT PERHITUNGAN 
    with tab3:
        st.markdown("## 📜 Riwayat Perhitungan SAW")
        if not st.session_state.history:
            st.info("Belum ada perhitungan yang disimpan.")
        else:
            if st.button("🗑️ Hapus Semua Riwayat"):
                st.session_state.history = []
                st.session_state.saved_results = {}
                st.rerun()
            for hist in st.session_state.history:
                with st.container():
                    col1, col2, col3 = st.columns([2,2,1])
                    with col1:
                        st.markdown(f"""
                        <div class="history-card">
                            <b>📌 {hist['name']}</b><br>
                            🕐 {hist['timestamp']}<br>
                            🏆 Pemenang: {hist['top_watch']['brand']} {hist['top_watch']['model']}<br>
                            📊 Score: {hist['top_watch']['score']:.4f}<br>
                            📱 Total: {hist['total_watches']}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        w = hist['weights']
                        st.markdown(f"""
                        <div class="history-card">
                            ⚙️ Bobot:<br>
                            📱 Ukuran Layar: {w['Display Size']:.3f}<br>
                            💧 Ketahanan Air: {w['Water Resistance']:.3f}<br>
                            🔋 Baterai: {w['Battery Life']:.3f}<br>
                            💰 Harga: {w['Price']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        if st.button("👁️ Detail", key=f"view_{hist['id']}"):
                            st.session_state.view_id = hist['id']
                        if st.button("🗑️ Hapus", key=f"del_{hist['id']}"):
                            idx = [i for i, h in enumerate(st.session_state.history) if h['id'] == hist['id']][0]
                            st.session_state.history.pop(idx)
                            if hist['id'] in st.session_state.saved_results:
                                del st.session_state.saved_results[hist['id']]
                            st.rerun()
                    st.markdown("---")
            # Tampilkan detail visualisasi jika ada view_id
            if 'view_id' in st.session_state and st.session_state.view_id in st.session_state.saved_results:
                detail_data = st.session_state.saved_results[st.session_state.view_id]
                st.markdown("### 🔍 Visualisasi dari Perhitungan Terseleksi")
                st.markdown(f"**Nama:** {next((h['name'] for h in st.session_state.history if h['id']==st.session_state.view_id), '')}")
                
                # Pie chart
                fig_pie = plot_pie_gradient(detail_data)
                st.pyplot(fig_pie)
                # Bar chart
                fig_bar = plot_bar_gradient(detail_data)
                st.pyplot(fig_bar)
                # Line chart
                fig_line = plot_line_chart_harga_saw(detail_data)
                st.pyplot(fig_line)
                
                if st.button("❌ Tutup Detail Visualisasi"):
                    del st.session_state.view_id
                    st.rerun()
    
    # VISUALISASI
    with tab4:
        st.markdown("## 📊 Visualisasi Data Smartwatch")
        result_filtered = st.session_state.saw_result
        if result_filtered is not None and not result_filtered.empty:
            # Pie chart
            fig_pie = plot_pie_gradient(result_filtered)
            st.pyplot(fig_pie)
            # Bar chart
            fig_bar = plot_bar_gradient(result_filtered)
            st.pyplot(fig_bar)
            # Line chart
            fig_line = plot_line_chart_harga_saw(result_filtered)
            st.pyplot(fig_line)
        else:
            st.info("🔍 Belum ada hasil. Klik tombol 'Hitung SAW' di sidebar.")
    
    with tab5:
        st.markdown("## 👥 Tentang Kelompok")
        st.markdown("Aplikasi **SmartWatchRank** dikembangkan sebagai proyek Sistem Pendukung Keputusan (SPK) menggunakan metode **Simple Additive Weighting (SAW)**.")

        st.markdown("""
        <style>
            /* Memilih elemen gambar yang dihasilkan oleh st.image di dalam kolom */
            .stColumn img {
                width: 150px !important;
                height: 150px !important;
                object-fit: cover !important;
                border-radius: 50% !important;
                border: 2px solid #c49a6c !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
                display: block !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # kolom profil anggota
        col1, col2 = st.columns(2, gap="large")

        with col1:
            try:
                st.image("titik.jpeg", width=180, caption="Titik Gemini")
            except:
                st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180, caption="Titik Gemini")
            st.markdown("**Nama:** Shientya Belva")
            st.markdown("**NIM:** 123240042")
            st.markdown("**Kontribusi:** Desain UI/UX, Analisis Data & Visualisasi")

        with col2:
            try:
                st.image("buncis.jpeg", width=180, caption="Buncis Bosok")
            except:
                st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180, caption="Buncis Bosok")
            st.markdown("**Nama:** Bylbiss El Haqqie")
            st.markdown("**NIM:** 123240003")
            st.markdown("**Peran:** Implementasi SAW")

        st.markdown("---")
        st.markdown("## 📖 Metode Simple Additive Weighting (SAW)")

        st.markdown("""
        **Simple Additive Weighting (SAW)** adalah salah satu metode paling populer dalam Sistem Pendukung Keputusan multi-kriteria. 
        Metode ini bekerja dengan melakukan **normalisasi** setiap kriteria terhadap nilai maksimum (untuk benefit) atau minimum (untuk cost), 
        kemudian mengalikannya dengan bobot yang telah ditentukan, dan menjumlahkan semua nilai untuk setiap alternatif.

        **Langkah-langkah SAW yang diterapkan dalam aplikasi ini:**

        1. **Menentukan kriteria**  
        - Ukuran Layar (Benefit) – semakin besar semakin baik  
        - Ketahanan Air (Benefit) – semakin tinggi rating kedalaman semakin baik  
        - Daya Tahan Baterai (Benefit) – semakin lama semakin baik  
        - Harga (Cost) – semakin murah semakin baik  

        2. **Memberikan bobot preferensi**  
        Pengguna dapat mengatur bobot melalui sidebar. Bobot akan otomatis dinormalisasi sehingga totalnya = 1.  

        3. **Normalisasi matriks keputusan**  
        - Kriteria *benefit* dinormalisasi dengan: `nilai / nilai_maksimum`  
        - Kriteria *cost* dinormalisasi dengan: `nilai_minimum / nilai`  

        4. **Menghitung skor akhir**  
        Setiap alternatif (smartwatch) dihitung skornya dengan menjumlahkan hasil perkalian nilai normalisasi dengan bobot masing-masing kriteria.  

        5. **Perankingan**  
        Smartwatch dengan skor SAW tertinggi menjadi rekomendasi terbaik.

        **Keunggulan SAW** adalah sederhana, mudah dipahami, dan cocok untuk kasus pemilihan produk dengan banyak kriteria seperti smartwatch.

        ---
        ### 🎯 Tujuan Aplikasi
        Membantu pengguna dalam memilih smartwatch yang paling sesuai dengan preferensi mereka secara objektif dan transparan.

        **SCPK 2025/2026** – Sistem Pendukung Keputusan Cerdas.
        """)

if __name__ == "__main__":
    main()