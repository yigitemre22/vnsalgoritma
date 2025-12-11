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
    CSV dosyasını okur ve virgülden (.)'e dönüştürerek sayısal kolonları float yapar.
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
# 2. BSM307 VNS SINIFI (Değişiklikler Buraya Uygulandı)
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
        node_df = pd.read_csv(node_file)
        
        # Düğüm Niteliklerini Grafiğe Ekle (node_id -> s_ms, r_node)
        for index, row in node_df.iterrows():
            node_id = int(row['node_id'])
            self.graph.add_node(node_id, 
                                processing_delay=row['s_ms'],   # İşlem Süresi (ms)
                                reliability=row['r_node'])      # Düğüm Güvenilirliği (r_node)

        # Kenar Verilerini Yükle
        edge_df = pd.read_csv(edge_file)
        
        # Kenar Niteliklerini Grafiğe Ekle
        for index, row in edge_df.iterrows():
            src = int(row['src'])
            dst = int(row['dst'])
            self.graph.add_edge(src, dst, 
                                capacity_mbps=row['capacity_mbps'], # Bant Genişliği (capacity_mbps)
                                delay_ms=row['delay_ms'],           # Gecikme (delay_ms)
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


    # =======================================================================
    # *** BAŞLANGIÇ ÇÖZÜMÜ GÜNCELLEMESİ (Değişkenliği Artırır) ***
    # =======================================================================
    def get_random_path(self, source, target, demand, max_length=150, max_tries=50):
        """
        Kapasite kısıtını sağlayan kenarları kullanarak kaynak ve hedef arasında 
        rastgele bir yürüyüş ile yol bulur.
        """
        # Hızlı erişim için geçerli komşuları önceden hesapla
        valid_successors = {
            u: [v for v in self.graph.successors(u) 
                if self.graph.has_edge(u, v) and self.graph.edges[u, v]['capacity_mbps'] >= demand]
            for u in self.graph.nodes
        }

        for _ in range(max_tries):
            path = [source]
            current_node = source
            
            for _ in range(max_length):
                
                if current_node == target:
                    return path

                successors = valid_successors.get(current_node, [])
                
                if not successors:
                    break # Çıkmaz sokak
                
                # Rastgele bir sonraki düğümü seç (geri döngüleri hafifçe engelle)
                candidates = [n for n in successors if n not in path[-2:]]
                
                if not candidates:
                    candidates = successors # Eğer geri döngüsüz yol yoksa, döngüye izin ver.

                next_node = random.choice(candidates)
                
                path.append(next_node)
                current_node = next_node

        return None # Yol bulunamadı


    def get_initial_solution(self, source, target, demand):
        """
        Başlangıç çözümü: Önce rastgele bir yol bulmaya çalışır, bulamazsa en kısa yolu dener.
        """
        # 1. Aşama: Rastgele Çözüm (Değişkenlik için)
        random_path = self.get_random_path(source, target, demand)
        if random_path:
            return random_path

        # 2. Aşama: Deterministik En Kısa Yol (Yedek olarak)
        print(f"Uyarı: Talep {demand} için rastgele yol bulunamadı, en kısa yol deneniyor.")
        valid_edges = [(u, v) for u, v, data in self.graph.edges(data=True) if data['capacity_mbps'] >= demand]
        temp_graph = nx.DiGraph(valid_edges)
        
        try:
            # Kenar sayısına göre en kısa yolu bulur
            return nx.shortest_path(temp_graph, source, target) 
        except nx.NetworkXNoPath:
            return None
    # =======================================================================
    # *** BAŞLANGIÇ ÇÖZÜMÜ GÜNCELLEMESİ SONU ***
    # =======================================================================

    # VNS Yardımcı Fonksiyonları (Talep (Demand) parametresi eklendi)
    def shaking(self, path, k, demand):
        new_path = copy.deepcopy(path)
        if len(new_path) < 4: return new_path

        # Sarsıntı noktalarını belirle
        i = random.randint(0, len(new_path) - 3)
        # k'yı kullanarak daha büyük aralıklar seçmeye çalış (min 2, max k+2)
        gap = random.randint(2, min(k + 2, len(new_path) - 1 - i)) 
        j = i + gap
        
        start_node = new_path[i]
        end_node = new_path[j]
        
        # Basitleştirilmiş Shaking: 
        # nx.all_simple_paths yerine, rastgele yürüyüş ile segmenti yeniden bul
        
        # Kapasite kısıtını sağlayan rastgele bir ara yol bul
        segment_path = self.get_random_path(start_node, end_node, demand, max_length=15, max_tries=10)
        
        if segment_path:
            # Segmentin ilk ve son düğümlerini koruyarak yolu birleştir
            final_path = new_path[:i] + segment_path + new_path[j+1:]
            
            # Oluşan yolun gerçekten Source'dan Target'a gittiğini kontrol etmeye gerek yok, 
            # çünkü bu operasyon sadece bir segmenti değiştirir.
            return final_path
            
        return new_path # Rastgele segment bulunamazsa orijinal yolu döndür

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

    def run_vns(self, source, target, demand, max_attempts=20):
        """VNS Algoritmasının Ana Döngüsü"""
        current_path = self.get_initial_solution(source, target, demand)
        
        if not current_path:
            return None, float('inf'), 0, 0, 0, 0 # Başlangıç yolu bile bulunamadı
        
        best_path = current_path
        best_cost, _, _, _, _ = self.calculate_metrics(best_path, demand)
        
        k_max = 10 # k_max'i artırarak daha büyük komşulukları keşfedebilirsiniz
        
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
    NODE_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_NodeData.csv"
    EDGE_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_EdgeData.csv"
    DEMAND_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_DemandData.csv"
    
    # VNS Motoru (Ağ Yükleniyor)
    vns_engine = BSM307VNS(NODE_FILE, EDGE_FILE)
    
    # Talep verilerini yükle
    demand_df = pd.read_csv(DEMAND_FILE)
    
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
            
            # Rastgeleliği maksimize etmek için her çalıştırmada random seed'i sıfırla
            # (Eğer tam olarak aynı donanım ve OS'de iseniz, bu, her çalıştırmanın birbirinden
            # farklı olmasını sağlamaz, ancak VNS'in farklı yollar bulma şansını artırır.)
            random.seed(None)
            
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
                print(f"   » Toplam Maliyet (Fitness): {cost:.4f}")
                print(f"   » Toplam Gecikme:           {delay:.2f} ms")
                print(f"   » Güvenilirlik:             %{reliability*100:.4f}")
                print(f"   » Yol: {path}") # Yeni yol çıktısı
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