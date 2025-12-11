import networkx as nx
import random
import math
import copy
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. VERİ OKUMA VE ÖN İŞLEME YARDIMCI FONKSİYONLARI
# ---------------------------------------------------------

def read_csv_with_comma_decimal(file_name, sep=';'):
    """
    CSV dosaasını okur ve virgülden (.)'e dönüştürerek sayısal kolonları float yapar.
    """
    df = pd.read_csv(file_name, sep=sep, encoding='utf-8')
    
    # Sayısal kolonları bul ve virgülleri noktaya çevir
    for col in df.columns:
        # String kolonlarda virgül kontrolü yap
        if df[col].dtype == 'object' and df[col].str.contains(',').any():
            # Virgülü nokta yap ve sayıya dönüştür
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            
    return df

# ---------------------------------------------------------
# 2. BSM307 VNS SINIFI (Veri Yüklenecek Şekilde Güncellendi)
# ---------------------------------------------------------

class BSM307VNS:
    def __init__(self, node_file, edge_file, w_delay=0.33, w_reliability=0.33, w_resource=0.34):
        self.w_delay = w_delay
        self.w_reliability = w_reliability
        self.w_resource = w_resource
        self.graph = nx.DiGraph()
        
        # Dosyalardan ağı yükle
        self.load_network(node_file, edge_file)
        self.num_nodes = len(self.graph.nodes)

    def load_network(self, node_file, edge_file):
        """Yüklenen Node ve Edge CSV'lerinden ağı NetworkX'e yükler."""
        print("Ağ verileri yükleniyor...")
        
        # Düğüm Verilerini Yükle
        node_df = read_csv_with_comma_decimal(node_file)
        
        # Düğüm Niteliklerini Grafiğe Ekle (node_id -> s_ms, r_node)
        for index, row in node_df.iterrows():
            node_id = int(row['node_id'])
            self.graph.add_node(node_id, 
                                processing_delay=row['s_ms'],  # İşlem Süresi (ms)
                                reliability=row['r_node'])     # Düğüm Güvenilirliği (r_node)

        # Kenar Verilerini Yükle
        edge_df = read_csv_with_comma_decimal(edge_file)
        
        # Kenar Niteliklerini Grafiğe Ekle
        for index, row in edge_df.iterrows():
            src = int(row['src'])
            dst = int(row['dst'])
            self.graph.add_edge(src, dst, 
                                capacity_mbps=row['capacity_mbps'], # Bant Genişliği (capacity_mbps)
                                delay_ms=row['delay_ms'],            # Gecikme (delay_ms)
                                reliability=row['r_link'])          # Bağlantı Güvenilirliği (r_link)
        
        print(f"Ağ yüklendi: {len(self.graph.nodes)} Düğüm, {len(self.graph.edges)} Kenar.")

    def calculate_metrics(self, path, demand_mbps):
        """
        Çok Amaçlı Maliyet ve Metrikleri hesaplar, Talep (Demand) kısıtını kontrol eder.
        """
        if not path or len(path) < 2:
            return float('inf'), 0, 0, 0, False # Geçersiz yol
        
        total_delay = 0
        reliability_log_cost = 0 
        resource_cost = 0
        real_reliability = 1.0

        # Düğümlerin maliyetleri
        for node in path[1:-1]:
            props = self.graph.nodes[node]
            total_delay += props['processing_delay']
            reliability_log_cost += -math.log(props['reliability'])
            real_reliability *= props['reliability']

        # Bağlantıların maliyetleri ve Kısıt Kontrolü
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not self.graph.has_edge(u, v):
                return float('inf'), 0, 0, 0, False
            
            edge_props = self.graph.edges[u, v]
            
            # *** Yeni Kısıt Kontrolü (Talep kısıtını karşılıyor mu?) ***
            if edge_props['capacity_mbps'] < demand_mbps:
                return float('inf'), 0, 0, 0, False # Uygun değil!
            
            # Metrik Hesaplamaları
            total_delay += edge_props['delay_ms']
            reliability_log_cost += -math.log(edge_props['reliability'])
            # Kaynak Maliyeti (1000 Mbps / Kapasite)
            resource_cost += (1000.0 / edge_props['capacity_mbps'])

        # Ağırlıklı Toplam Maliyet
        total_fitness = (self.w_delay * total_delay) + \
                        (self.w_reliability * reliability_log_cost) + \
                        (self.w_resource * resource_cost)
                        
        return total_fitness, total_delay, real_reliability, resource_cost, True # True: Uygundur

    def get_initial_solution(self, source, target, demand):
        """Başlangıç çözümü: Sadece kapasite kısıtını sağlayan en kısa yolu bulur."""
        
        # Geçerli kenarları filtrele: Sadece talebi karşılayanları tut.
        valid_edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data['capacity_mbps'] >= demand]
        
        # Yeni bir geçici alt grafik oluştur
        temp_graph = nx.DiGraph(valid_edges)
        
        try:
            # Geçici grafikte en kısa yolu bul
            return nx.shortest_path(temp_graph, source, target) 
        except nx.NetworkXNoPath:
            return None

    # VNS Yardımcı Fonksiyonları (Talep (Demand) parametresi eklendi)
    def shaking(self, path, k, demand):
        # ... (Önceki VNS sarsıntı mantığı aynı kalır, ama alt-yol ararken demand'ı gözetmeli)
        # Bu kısım kod karmaşıklığını artırdığı için basit shaking kullanılmıştır.
        # Basitlik için sadece yolu sarsıp fitness'ı sonradan kontrol edeceğiz.
        
        new_path = copy.deepcopy(path)
        if len(new_path) < 4: return new_path

        # Sarsıntı mantığı (rastgele bir segmenti yeniden rotalama)
        i = random.randint(0, len(new_path) - 3)
        gap = random.randint(2, min(5, len(new_path) - 1 - i))
        j = i + gap
        
        # Geçerli kenarları filtrele (sadece talep kısıtını sağlayanlar)
        valid_edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data['capacity_mbps'] >= demand]
        temp_graph = nx.DiGraph(valid_edges)
        
        try:
            # u ve v arasında rastgele bir yol bul
            alternatives = list(nx.all_simple_paths(temp_graph, new_path[i], new_path[j], cutoff=gap+2))
            if alternatives:
                chosen_segment = random.choice(alternatives)
                final_path = new_path[:i] + chosen_segment + new_path[j+1:]
                return final_path
        except:
            pass
            
        return new_path

    def local_search(self, path, demand):
        # ... (Önceki Local Search mantığı aynı kalır, ama fitness kontrolü demand'ı kullanır)
        best_path = path
        best_cost, _, _, _, is_feasible = self.calculate_metrics(path, demand)
        
        if not is_feasible: return path # Zaten uygun değilse iyileştirme yapamaz

        # Kestirme denemesi (bir düğümü atlama)
        if len(path) > 3:
            for _ in range(5): 
                idx = random.randint(1, len(path)-2)
                # Kestirme denenecekse, bu kenarın da demand'ı karşılaması lazım.
                if self.graph.has_edge(path[idx-1], path[idx+1]) and \
                   self.graph.edges[path[idx-1], path[idx+1]]['capacity_mbps'] >= demand:
                    
                    neighbor = path[:idx] + path[idx+1:]
                    cost, _, _, _, is_feasible = self.calculate_metrics(neighbor, demand)
                    
                    if is_feasible and cost < best_cost:
                        best_cost = cost
                        best_path = neighbor
        return best_path

    def run_vns(self, source, target, demand, max_attempts=100):
        """VNS Algoritmasının Ana Döngüsü"""
        current_path = self.get_initial_solution(source, target, demand)
        
        if not current_path:
            return None, float('inf'), 0, 0, 0, 0 # Başlangıç yolu bile bulunamadı
        
        best_path = current_path
        best_cost, _, _, _, _ = self.calculate_metrics(best_path, demand)
        
        k_max = 2
        
        for iteration in range(max_attempts):
            k = 1
            while k <= k_max:
                shaken_path = self.shaking(best_path, k, demand)
                improved_path = self.local_search(shaken_path, demand)
                
                cost, delay, reliability, resource, is_feasible = self.calculate_metrics(improved_path, demand)
                
                if is_feasible and cost < best_cost:
                    best_path = improved_path
                    best_cost = cost
                    k = 1 
                else:
                    k += 1 
            
        # Son metrikleri hesapla
        final_cost, final_delay, final_reliability, final_resource, is_feasible = self.calculate_metrics(best_path, demand)
        
        if not is_feasible: # Kontrol: VNS sonunda yol geçersiz kalmışsa
             return None, float('inf'), 0, 0, 0, 0
             
        return best_path, final_cost, final_delay, final_reliability, final_resource, len(best_path)


# ---------------------------------------------------------
# 3. ANA ÇALIŞTIRMA VE DENEY MODÜLÜ
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Proje verilerinden motoru başlat
    NODE_FILE = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    EDGE_FILE = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    DEMAND_FILE = "BSM307_317_Guz2025_TermProject_DemandData.csv"
    
    # VNS Motoru (Ağ Yükleniyor)
    vns_engine = BSM307VNS(NODE_FILE, EDGE_FILE)
    
    # Talep verilerini yükle
    demand_df = read_csv_with_comma_decimal(DEMAND_FILE)
    
    results = []
    
    print("\n" + "="*90)
    print(f"VERİYE DAYALI VNS ALGORİTMASI - TOPLAM {len(demand_df)} TALEP ÇİFTİ ÜZERİNDE DENEY")
    print("="*90)

    # 2. Tüm talep çiftleri üzerinde döngü
    for index, row in demand_df.iterrows():
        S = int(row['src'])
        D = int(row['dst'])
        demand = row['demand_mbps']
        
        # Her bir talep çifti için 5 kere tekrar (Proje kuralına uygunluk için)
        for run_id in range(1, 6):
            
            print(f"\n--- TALEP #{index+1} ({S} -> {D} | {demand} Mbps) | Tekrar #{run_id} ---")
            
            # 3. VNS'i Çalıştır
            path, cost, delay, reliability, resource_cost, length = vns_engine.run_vns(S, D, demand, max_attempts=50)
            
            # Sonuçları kaydet
            results.append({
                'Talep ID': index+1,
                'Kaynak': S,
                'Hedef': D,
                'Demand (Mbps)': demand,
                'Tekrar': run_id,
                'Maliyet (Fitness)': cost,
                'Gecikme (ms)': delay,
                'Güvenilirlik (%)': reliability * 100,
                'Kaynak Maliyeti': resource_cost,
                'Adım Sayısı': length
            })
            
            # Terminale çıktı yazdır
            if path and cost != float('inf'):
                print(f"✅ Rota Bulundu ({length} Adım):")
                print(f"   » Toplam Maliyet (Fitness): {cost:.4f}")
                print(f"   » Toplam Gecikme:           {delay:.2f} ms")
                print(f"   » Güvenilirlik:             %{reliability*100:.4f}")
            else:
                print("❌ Geçerli ve kapasite kısıtını sağlayan rota bulunamadı.")


    # 4. Genel Özet Tablosu
    results_df = pd.DataFrame(results)
    
    # Başarısız olanları ayır
    successful_runs = results_df[results_df['Maliyet (Fitness)'] != float('inf')]
    
    print("\n\n" + "="*90)
    print("TOPLU DENEY SONUÇLARI ÖZETİ (Tüm Tekrarlar Dahil)")
    print("="*90)
    print(successful_runs.to_string())
    
    # Raporlama için faydalı olacak Ortalama Değerler
    avg_results = successful_runs.groupby(['Talep ID', 'Kaynak', 'Hedef', 'Demand (Mbps)']).agg({
        'Maliyet (Fitness)': 'mean',
        'Gecikme (ms)': 'mean',
        'Güvenilirlik (%)': 'mean',
        'Adım Sayısı': 'mean'
    }).reset_index()
    
    print("\n\n" + "="*90)
    print("TALEP BAŞINA ORTALAMA SONUÇLAR (Raporlama İçin)")
    print("="*90)
    print(avg_results.round(4).to_string())