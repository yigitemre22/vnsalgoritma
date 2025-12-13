import sys
import networkx as nx
import numpy as np
import random
import math
import time
import csv
import os
import pandas as pd
from collections import Counter

# PyQt6 ve Matplotlib importları
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QProgressBar
)

# ==========================================
# 1. ARKA PLAN MANTIĞI (VNS ALGORİTMASI)
# ==========================================

# Dosya yolları. Programın çalıştığı dizinde olmaları beklenir.
NODE_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_NodeData.csv"
EDGE_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_EdgeData.csv"
DEMAND_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_DemandData.csv"
INF = float('inf')


def read_csv_robust(file_name, sep=','):
    """CSV okur, başlıkları ve ondalık ayırıcıları temizler."""
    # Dosya path'ini düzgün yönetmek için os.path.join kullanıldı
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    
    try:
        df = pd.read_csv(full_path, sep=sep)
        df.columns = df.columns.str.strip() 
        for col in df.columns:
            if df[col].dtype == 'object':
                # Virgülü ondalık ayırıcı olarak kabul et
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass
        return df
    except FileNotFoundError:
        print(f"HATA: {file_name} bulunamadı!")
        return None


def create_graph_from_csv(node_file, edge_file):
    """CSV Dosyalarından Yönlü Graf (DiGraph) Oluşturur."""
    G = nx.DiGraph() 
    node_df = read_csv_robust(node_file)
    edge_df = read_csv_robust(edge_file)

    if node_df is None or edge_df is None:
        return G # Boş graf döndür

    # Düğümleri Yükle
    G.add_nodes_from([
        (row['node_id'], {'processing_delay': row['s_ms'], 'reliability': row['r_node']})
        for _, row in node_df.iterrows()
    ])

    # Kenarları Yükle (Çift Yönlü Bağlantı Sağlanır)
    for _, row in edge_df.iterrows():
        try:
            u, v = int(row['src']), int(row['dst'])
            link_attrs = {
                'capacity_mbps': row['capacity_mbps'],
                'delay_ms': row['delay_ms'],
                'reliability': row['r_link']
            }
            # Çift yönlü kenar ekleme (G.has_edge kontrolü DiGraph'ta gereksiz)
            G.add_edge(u, v, **link_attrs)
            G.add_edge(v, u, **link_attrs)
        except Exception:
            continue
            
    # Grafiğin bağlı olup olmadığını kontrol etmek DiGraph'ta daha karmaşıktır (Strongly connected)
    # Basitçe yüklenen DiGraph'u döndürelim.
    return G

# VNS Sınıfı
class BSM307VNS:
    def __init__(self, graph, w_delay=None, w_reliability=None, w_resource=None):
        self.graph = graph
        
        # Ağırlıklar belirtilmemişse, rastgele atanır
        if w_delay is None:
            w = np.random.rand(3)
            w /= w.sum()
            self.w_delay = w[0]
            self.w_reliability = w[1] 
            self.w_resource = w[2]
        else:
            self.w_delay = w_delay
            self.w_reliability = w_reliability
            self.w_resource = w_resource

    def _check_max_node_visits(self, path, max_visits=5):
        """Bir yoldaki her düğümün ziyaret sayısını kontrol eder."""
        return max(Counter(path).values()) <= max_visits

    def _get_valid_subgraph(self, demand):
        """Kapasite kısıtlamasına uyan kenarlarla geçici bir alt grafik oluşturur."""
        valid_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if d.get('capacity_mbps', 0) >= demand
        ]
        return nx.DiGraph(valid_edges)
        
    def calculate_metrics(self, path, demand, max_visits=5):
        """Yolun metriklerini hesaplar ve tek amaçlı maliyet fonksiyonunu döndürür."""
        if not path or len(path) < 2 or (len(path) > max_visits * 2 and not self._check_max_node_visits(path, max_visits)):
            return INF, 0, 0, 0, False

        total_delay, log_rel, resource, real_rel = 0, 0, 0, 1.0

        try:
            # Düğüm metrikleri (Sadece ara düğümlerin işlem gecikmesini ekle)
            for node in path[1:-1]:
                n = self.graph.nodes[node]
                total_delay += n['processing_delay']
                log_rel += -math.log(n['reliability'])
                real_rel *= n['reliability']
            
            # Kenar metrikleri
            for u, v in zip(path[:-1], path[1:]):
                e = self.graph.edges[u, v]
                if e['capacity_mbps'] < demand: return INF, 0, 0, 0, False

                total_delay += e['delay_ms']
                log_rel += -math.log(e['reliability'])
                resource += 1000.0 / e['capacity_mbps']
                real_rel *= e['reliability']

        except (KeyError, ValueError, ZeroDivisionError): 
            return INF, 0, 0, 0, False

        # Tek Amaçlı Maliyet Fonksiyonu
        cost = (self.w_delay * total_delay + self.w_reliability * log_rel + self.w_resource * resource)

        return cost, total_delay, real_rel, resource, True
    
    def get_initial_solution(self, source, target, demand):
        """Kapasiteye uyan grafikte en kısa yolu bulur."""
        temp_graph = self._get_valid_subgraph(demand)
        try:
            return nx.shortest_path(temp_graph, source, target, weight='delay_ms')
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None

    def shaking(self, path, demand):
        """Yolun rastgele bir bölümünü keser ve yerine en kısa yolu koyar."""
        if len(path) < 4: return path
        
        i = random.randint(0, len(path) - 3)
        j = random.randint(i + 2, len(path) - 1)
        start_node, end_node = path[i], path[j]
        
        temp_graph = self._get_valid_subgraph(demand)
        
        try:
            path_to_insert = nx.shortest_path(temp_graph, start_node, end_node, weight='delay_ms')
            return path[:i] + path_to_insert + path[j + 1:]
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return path

    def local_search(self, path, demand):
        """Yerel iyileştirme adımı: Tek bir düğümü atlamayı dener."""
        best_path = list(path)
        best_cost, *_ , feasible = self.calculate_metrics(best_path, demand)
        if not feasible: return path
        
        improved = True
        while improved:
            improved = False
            for idx in range(1, len(best_path) - 1):
                u, v = best_path[idx-1], best_path[idx+1]
                
                if self.graph.has_edge(u, v) and self.graph.edges[u, v].get('capacity_mbps', 0) >= demand:
                    candidate = best_path[:idx] + best_path[idx+1:]
                    cost, *_ , feasible = self.calculate_metrics(candidate, demand)

                    if feasible and cost < best_cost:
                        best_path, best_cost = candidate, cost
                        improved = True
                        break
        return best_path

    def run_vns(self, source, target, demand, max_attempts=50, k_max=4):
        """VNS ana döngüsü."""
        start_time = time.time()
        path = self.get_initial_solution(source, target, demand)
        cost, *_ , feasible = self.calculate_metrics(path, demand)
        
        if not feasible: 
            elapsed = (time.time() - start_time) * 1000
            return None, INF, 0, 0, 0, 0, elapsed

        best_path, best_cost, current_path = path, cost, path

        for _ in range(max_attempts):
            k = 1
            while k <= k_max:
                shaken = self.shaking(current_path, demand)
                improved = self.local_search(shaken, demand)
                cost, *_ , feasible = self.calculate_metrics(improved, demand)

                if feasible and cost < best_cost:
                    best_path, best_cost, current_path = improved, cost, improved
                    k = 1
                else:
                    k += 1

            current_path = best_path
        
        cost, delay, rel, res, _ = self.calculate_metrics(best_path, demand)
        elapsed = (time.time() - start_time) * 1000
        return best_path, cost, delay, rel, res, len(best_path), elapsed

# ==========================================
# 2. ARAYÜZ (PYQT6) - VNS UYARLANMIŞ
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - QoS VNS Projesi (CSV Verileri)")
        self.resize(1300, 900)

        # CSV Dosyalarından Grafi Yükle (DiGraph)
        self.G = create_graph_from_csv(NODE_FILE, EDGE_FILE)
        self.node_count = self.G.number_of_nodes()
        
        # Konumlandırma
        self.pos = nx.spring_layout(self.G, seed=42)

        # Ana Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Tekli Çalıştırma & Görselleştirme
        self.tab1 = QWidget()
        self.init_single_run_tab()
        self.tabs.addTab(self.tab1, "🔍 Tekli Analiz & Görselleştirme")

        # Tab 2: Toplu Deney & İstatistik
        self.tab2 = QWidget()
        self.init_batch_test_tab()
        self.tabs.addTab(self.tab2, "📊 Toplu Deney (DemandData.csv)")

    # ---------------------------------------------------
    # TAB 1: TEKLİ ÇALIŞTIRMA
    # ---------------------------------------------------
    def init_single_run_tab(self):
        layout = QHBoxLayout(self.tab1)

        # Sol Panel
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_panel.setStyleSheet("background-color: black; border-right: 1px solid #ddd;")
        l_layout = QVBoxLayout(left_panel)

        l_layout.addWidget(QLabel("<h2>VNS Ayarları</h2>"))
        l_layout.addWidget(QLabel(f"Toplam Düğüm Sayısı: {self.node_count}"))
        
        # Source/Destination SpinBoxes
        l_layout.addWidget(QLabel("Kaynak (S):"))
        self.spin_s = QSpinBox(); self.spin_s.setRange(0, self.node_count * 2); self.spin_s.setValue(0)
        l_layout.addWidget(self.spin_s)

        l_layout.addWidget(QLabel("Hedef (D):"))
        self.spin_d = QSpinBox(); self.spin_d.setRange(0, self.node_count * 2); self.spin_d.setValue(10)
        l_layout.addWidget(self.spin_d)

        l_layout.addWidget(QLabel("Min. Bant Genişliği (B):"))
        self.spin_bw = QSpinBox(); self.spin_bw.setRange(0, 10000); self.spin_bw.setSuffix(" Mbps")
        self.spin_bw.setValue(100) # Varsayılan değer
        l_layout.addWidget(self.spin_bw)

        l_layout.addWidget(QLabel("<h3>Amaç Ağırlıkları (Wd + Wr + Wres = 1.0)</h3>"))
        self.spin_wd = QDoubleSpinBox(); self.spin_wd.setValue(0.33); self.spin_wd.setSingleStep(0.01); self.spin_wd.setDecimals(2)
        l_layout.addWidget(QLabel("W_Delay:")); l_layout.addWidget(self.spin_wd)
        
        self.spin_wr = QDoubleSpinBox(); self.spin_wr.setValue(0.33); self.spin_wr.setSingleStep(0.01); self.spin_wr.setDecimals(2)
        l_layout.addWidget(QLabel("W_Rel:")); l_layout.addWidget(self.spin_wr)

        self.spin_wres = QDoubleSpinBox(); self.spin_wres.setValue(0.34); self.spin_wres.setSingleStep(0.01); self.spin_wres.setDecimals(2)
        l_layout.addWidget(QLabel("W_Res:")); l_layout.addWidget(self.spin_wres)

        l_layout.addWidget(QLabel("<h3>VNS Parametreleri</h3>"))
        l_layout.addWidget(QLabel("Max İterasyon Sayısı (max_attempts):"))
        self.spin_max_attempts = QSpinBox(); self.spin_max_attempts.setValue(50); self.spin_max_attempts.setRange(10, 500)
        l_layout.addWidget(self.spin_max_attempts)
        
        # Fazladan boşluk bırak
        l_layout.addStretch()

        self.btn_run = QPushButton("🚀 VNS Çalıştır")
        self.btn_run.setStyleSheet("background-color: #007bff; color: black; padding: 10px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_single)
        l_layout.addWidget(self.btn_run)

        self.txt_output = QTextEdit(); self.txt_output.setReadOnly(True)
        l_layout.addWidget(self.txt_output)
        
        layout.addWidget(left_panel)

        # Sağ Panel (Matplotlib)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # İlk çizim
        if self.node_count > 0:
             s_def = next(iter(self.G.nodes()))
             d_def = list(self.G.nodes())[-1]
             self.plot_graph(None, s_def, d_def)

    def run_single(self):
        S = self.spin_s.value()
        D = self.spin_d.value()
        B = self.spin_bw.value()
        Wd, Wr, Wres = self.spin_wd.value(), self.spin_wr.value(), self.spin_wres.value()
        max_attempts = self.spin_max_attempts.value()

        if S not in self.G or D not in self.G:
            self.txt_output.setText(f"HATA: {S} veya {D} düğümü grafikte yok.")
            return

        self.txt_output.setText("VNS Hesaplanıyor...")
        QApplication.processEvents()
        
        # VNS Motorunu oluştur (Belirtilen sabit ağırlıklarla)
        vns_engine = BSM307VNS(self.G, Wd, Wr, Wres)

        path, cost, delay, rel, res, steps, time_ms = vns_engine.run_vns(S, D, B, max_attempts=max_attempts)

        if path and cost != INF:
            rel_percent = rel * 100
            
            # Yolun gösterimi (İlk 3 ve son 3 düğüm)
            path_str = ' -> '.join(map(str, path[:3]))
            if steps > 6:
                path_str += f" -> ... ({steps-1} kenar) -> " + ' -> '.join(map(str, path[-3:]))
            else:
                path_str = ' -> '.join(map(str, path))
            
            msg = (f"✅ SONUÇ:\nSüre: {time_ms:.2f} ms\nMaliyet (Wd={Wd:.2f}, Wr={Wr:.2f}, Wres={Wres:.2f}): {cost:.4f}\n"
                   f"Hop Sayısı: {steps-1}\n\n"
                   f"Yol: {path_str}\n\n"
                   f"Metrikler:\n"
                   f"  Toplam Gecikme: {delay:.2f} ms\n"
                   f"  Genel Güvenilirlik: %{rel_percent:.4f}\n"
                   f"  Kaynak Kullanımı: {res:.2f}")
            
            self.txt_output.setText(msg)
            self.plot_graph(path, S, D)
        else:
            self.txt_output.setText("❌ Yol Bulunamadı!\nBu bant genişliği veya topoloji kısıtlarından kaynaklanabilir.")
            self.plot_graph(None, S, D)

    def plot_graph(self, path, S, D):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Genel Düğümleri ve Kenarları Çiz (Soluk)
        other_nodes = [n for n in self.G.nodes() if n != S and n != D and (not path or n not in path)]
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=other_nodes, 
                               node_size=15, node_color='lightgray', alpha=0.2)
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.05, edge_color='gray')
        
        # Kaynak ve Hedef Vurgula
        if S in self.G and D in self.G:
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[S, D], 
                                   node_color=['blue', 'red'], node_size=120)
            labels_sd = {S: str(S), D: str(D)}
            nx.draw_networkx_labels(self.G, self.pos, labels_sd, ax=ax, 
                                   font_size=6, font_color='black', font_weight='bold')
        
        if path:
            # Yolu çiz
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=edges, 
                                   edge_color='green', width=3, alpha=0.8)
            
            # Yoldaki ara düğümler
            path_nodes_mid = [n for n in path if n != S and n != D]
            if path_nodes_mid:
                nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=path_nodes_mid, 
                                       node_size=80, node_color='lime', alpha=0.9)
                labels_path = {node: str(node) for node in path_nodes_mid}
                nx.draw_networkx_labels(self.G, self.pos, labels_path, ax=ax, 
                                       font_size=5, font_color='black')
        
        ax.set_title(f"Topoloji (S={S} -> D={D})")
        ax.axis('off')
        self.canvas.draw()

    # ---------------------------------------------------
    # TAB 2: TOPLU DENEY MODÜLÜ (VNS UYARLANDI)
    # ---------------------------------------------------
    def load_demands(self):
        """CSV'den talepleri oku (Pandas ile daha güvenilir)"""
        df = read_csv_robust(DEMAND_FILE)
        if df is None: return []
        
        demands = []
        for _, row in df.iterrows():
            try:
                # Verilerin doğru tiplere dönüştürüldüğünden emin ol
                s = int(row['src'])
                d = int(row['dst'])
                bw = float(row['demand_mbps'])
                demands.append((s, d, bw))
            except:
                continue
        return demands

    def init_batch_test_tab(self):
        layout = QVBoxLayout(self.tab2)

        # Üst Kontrol Paneli
        top_panel = QFrame()
        top_panel.setStyleSheet("background-color: #e9ecef; border-radius: 5px;")
        h_layout = QHBoxLayout(top_panel)

        h_layout.addWidget(QLabel("<b>DemandData.csv Kullanılarak Test Yapılacak</b>"))
        
        h_layout.addWidget(QLabel("Tekrar Sayısı (Her Talep İçin - Random Ağırlıklarla):"))
        self.spin_repeat_count = QSpinBox(); self.spin_repeat_count.setValue(5); self.spin_repeat_count.setRange(1, 20)
        h_layout.addWidget(self.spin_repeat_count)

        self.btn_start_batch = QPushButton("🧪 CSV İLE VNS DENEYİNİ BAŞLAT")
        self.btn_start_batch.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.btn_start_batch.clicked.connect(self.run_batch_experiment)
        h_layout.addWidget(self.btn_start_batch)
        
        self.btn_export = QPushButton("💾 Sonuçları Kaydet")
        self.btn_export.clicked.connect(self.export_csv)
        h_layout.addWidget(self.btn_export)

        layout.addWidget(top_panel)

        # İlerleme Çubuğu
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Tablo
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Senaryo (ID)", "S -> D", "Talep (BW)", "Başarı Oranı (%)", 
            "Ort. Maliyet", "Std. Sapma", "En İyi Cost", "En Kötü Cost", "Ort. Süre (ms)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        # Sonuç verisini depolamak için
        self.batch_results_data = []

    def run_batch_experiment(self):
        demands = self.load_demands()
        
        if not demands:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("HATA: DemandData.csv Okunamadı veya Boş."))
            return

        repeats = self.spin_repeat_count.value()

        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(demands))
        self.batch_results_data = [] # Yeni sonuç listesi

        # Deneyleri Çalıştır
        for i, (S, D, B) in enumerate(demands):
            all_costs = []
            all_times = []
            success_count = 0
            best_run_cost = INF
            best_run_metrics = {}
            
            # Bu talep için 'repeats' kadar çalıştır (Her seferinde rastgele ağırlıklar)
            for _ in range(repeats):
                # VNS motorunu her tekrarda yeni, rastgele ağırlıklarla oluştur
                vns_engine = BSM307VNS(self.G) 
                
                path, cost, delay, rel, res, steps, t = vns_engine.run_vns(S, D, B, max_attempts=50) 
                
                if path and cost != INF:
                    all_costs.append(cost)
                    all_times.append(t)
                    success_count += 1
                    
                    if cost < best_run_cost:
                        best_run_cost = cost
                        best_run_metrics = {
                            'Path': ' -> '.join(map(str, path)),
                            'W_Delay': vns_engine.w_delay,
                            'W_Rel': vns_engine.w_reliability,
                            'W_Res': vns_engine.w_resource,
                            'Delay': delay,
                            'Reliability': rel,
                            'ResourceUsage': res
                        }
            
            # İstatistik Hesapla
            if all_costs:
                avg_cost = np.mean(all_costs)
                std_dev = np.std(all_costs)
                best_cost = np.min(all_costs)
                worst_cost = np.max(all_costs)
                avg_time = np.mean(all_times)
            else:
                avg_cost = std_dev = best_cost = worst_cost = avg_time = 0

            success_rate = (success_count / repeats) * 100

            # Tabloya Ekle
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(i+1)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{S} -> {D}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{B:.0f} Mbps"))
            self.table.setItem(row, 3, QTableWidgetItem(f"%{success_rate:.0f}"))
            
            if success_count > 0:
                self.table.setItem(row, 4, QTableWidgetItem(f"{avg_cost:.4f}"))
                self.table.setItem(row, 5, QTableWidgetItem(f"{std_dev:.4f}"))
                self.table.setItem(row, 6, QTableWidgetItem(f"{best_cost:.4f}"))
                self.table.setItem(row, 7, QTableWidgetItem(f"{worst_cost:.4f}"))
                self.table.setItem(row, 8, QTableWidgetItem(f"{avg_time:.2f}"))
                
                # Kayıt için ayrıntılı veriyi sakla
                self.batch_results_data.append({
                    'ScenarioID': i + 1,
                    'Source': S,
                    'Destination': D,
                    'Demand_Mbps': B,
                    'SuccessRate': f"{success_rate:.0f}%",
                    'Avg_Cost': f"{avg_cost:.4f}",
                    'Std_Dev': f"{std_dev:.4f}",
                    'Best_Cost': f"{best_cost:.4f}",
                    'Worst_Cost': f"{worst_cost:.4f}",
                    'Avg_Time_ms': f"{avg_time:.2f}",
                    **best_run_metrics
                })
            else:
                self.table.setItem(row, 4, QTableWidgetItem("BAŞARISIZ"))
                for c in range(5, 9): self.table.setItem(row, c, QTableWidgetItem("-"))
                
                self.batch_results_data.append({
                    'ScenarioID': i + 1, 'Source': S, 'Destination': D, 'Demand_Mbps': B,
                    'SuccessRate': "0%", 'Avg_Cost': "INF", 'Std_Dev': "-", 
                    'Best_Cost': "INF", 'Worst_Cost': "INF", 'Avg_Time_ms': "-",
                    'Path': "No Path Found"
                })

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

    def export_csv(self):
        """Toplu deney sonuçlarını ayrıntılı olarak kaydeder."""
        if not self.batch_results_data:
            print("Uyarı: Kaydedilecek toplu sonuç bulunamadı.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Toplu VNS Sonuçlarını Kaydet", "VNS_Batch_Results.csv", "CSV Files (*.csv)")
        if path:
            keys = self.batch_results_data[0].keys()
            with open(path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.batch_results_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())