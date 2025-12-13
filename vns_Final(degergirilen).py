import csv
import math
import random
import time
import copy
import os
from collections import deque

# --- AYARLAR VE SABİTLER ---
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_RESOURCE = 0.34
MAX_BANDWIDTH_MBPS = 1000.0
# Algoritma parametreleri
MAX_VNS_ITER = 20  
K_MAX = 3          

class NetworkGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def load_data(self, node_file, edge_file):
        """ Ağ bağlantılarını çift yönlü yapar ve verileri yükler. """
        
        # --- DÜĞÜM (NODE) VERİLERİNİ YÜKLE ---
        try:
            with open(node_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]

                for row in reader:
                    try:
                        nid = int(row['node_id'])
                        self.nodes[nid] = {
                            's_ms': float(row['s_ms']),
                            'r_node': float(row['r_node'])
                        }
                        if nid not in self.edges:
                            self.edges[nid] = {}
                    except Exception:
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"HATA: Düğüm dosyası bulunamadı: {node_file}")

        # --- KENAR (EDGE) VERİLERİNİ YÜKLE VE TERSİNİ EKLE (ÇİFT YÖNLÜ DÜZELTME) ---
        raw_edges = []
        try:
            with open(edge_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]

                for row in reader:
                    try:
                        raw_edges.append((
                            int(row['src']), 
                            int(row['dst']),
                            {
                                'bw': float(row['capacity_mbps']),
                                'delay': float(row['delay_ms']),
                                'r_link': float(row['r_link'])
                            }
                        ))
                    except Exception:
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"HATA: Kenar dosyası bulunamadı: {edge_file}")

        # Orijinal ve Ters Yönleri Ekle
        for u, v, props in raw_edges:
            # u -> v
            if u not in self.edges: self.edges[u] = {}
            self.edges[u][v] = props

            # v -> u (Çift yönlü bağlantı varsayımı)
            if v not in self.edges: self.edges[v] = {}
            if u not in self.edges[v]:
                 self.edges[v][u] = props 
                 
    def calculate_path_metrics(self, path):
        """ BSM307 proje formüllerine göre rotanın metriklerini hesaplar. """
        if not path or len(path) < 2:
            return float('inf'), float('inf'), float('inf'), float('inf')

        total_link_delay = 0.0
        total_proc_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        dest_node = path[-1]

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            if u not in self.edges or v not in self.edges[u] or v not in self.nodes:
                return float('inf'), float('inf'), float('inf'), float('inf')

            edge = self.edges[u][v]
            node_v = self.nodes[v]

            total_link_delay += edge['delay']
            reliability_cost += -math.log(edge['r_link']) if edge['r_link'] > 0 else 100
            resource_cost += (MAX_BANDWIDTH_MBPS / edge['bw'])

            if v != dest_node:
                total_proc_delay += node_v['s_ms']
                reliability_cost += -math.log(node_v['r_node']) if node_v['r_node'] > 0 else 100

        total_delay = total_link_delay + total_proc_delay
        weighted_cost = (W_DELAY * total_delay) + \
                        (W_RELIABILITY * reliability_cost) + \
                        (W_RESOURCE * resource_cost)

        return weighted_cost, total_delay, math.exp(-reliability_cost), resource_cost

class VNS_Optimizer:
    def __init__(self, graph):
        self.graph = graph

    def get_initial_solution(self, src, dst):
        """ Random BFS ile yol arama (Yol bulma garantili, stokastik başlangıç). """
        queue = deque([(src, [src])])
        visited = {src}
        
        while queue:
            current, path = queue.popleft()
            
            if current == dst:
                return path
            
            neighbors = list(self.graph.edges.get(current, {}).keys())
            random.shuffle(neighbors) # Stokastik başlangıç
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
        return None

    def shake(self, path, k):
        """ VNS Çalkalama Operatörü. """
        if len(path) < 4: return path
        new_path = copy.deepcopy(path)
        
        idx1 = random.randint(1, len(new_path) - 2)
        max_gap = min(len(new_path)-1, idx1 + k + 1)
        idx2 = random.randint(idx1 + 1, max_gap)
        
        node_a = new_path[idx1-1]
        node_b = new_path[idx2]
        
        temp_visited = set(new_path[:idx1])
        sub_path = []
        
        def random_bridge(curr):
            if curr == node_b: return True
            if len(sub_path) > 8: return False
            
            neighbors = list(self.graph.edges.get(curr, {}).keys())
            random.shuffle(neighbors)
            
            for n in neighbors:
                if n == node_b: return True
                if n not in temp_visited:
                    temp_visited.add(n)
                    sub_path.append(n)
                    if random_bridge(n): return True
                    sub_path.pop()
                    temp_visited.remove(n)
            return False

        if random_bridge(node_a):
            return new_path[:idx1] + sub_path + new_path[idx2:]
        return path

    def local_search(self, path):
        """ Yerel Arama: Basit kestirme (Shortcut) yöntemi. """
        best_path = copy.deepcopy(path)
        best_cost, _, _, _ = self.graph.calculate_path_metrics(best_path)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(best_path) - 2):
                for j in range(i + 2, len(best_path)):
                    u = best_path[i]
                    v = best_path[j]
                    
                    if u in self.graph.edges and v in self.graph.edges.get(u, {}):
                        new_proposal = best_path[:i+1] + best_path[j:]
                        new_cost, _, _, _ = self.graph.calculate_path_metrics(new_proposal)
                        
                        if new_cost < best_cost:
                            best_path = new_proposal
                            best_cost = new_cost
                            improved = True
                            break
                if improved: break
        return best_path

    def run(self, src, dst, max_iter=MAX_VNS_ITER, k_max=K_MAX):
        """ VNS Ana Döngüsü """
        current_path = self.get_initial_solution(src, dst)
        if not current_path: return None, float('inf'), {}
        
        current_cost, _, _, _ = self.graph.calculate_path_metrics(current_path)
        best_path = current_path
        best_cost = current_cost
        
        for _ in range(max_iter):
            k = 1
            while k <= k_max:
                shaken_path = self.shake(current_path, k)
                improved_path = self.local_search(shaken_path)
                improved_cost, _, _, _ = self.graph.calculate_path_metrics(improved_path)
                
                if improved_cost < current_cost:
                    current_path = improved_path
                    current_cost = improved_cost
                    k = 1
                else:
                    k += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = current_path

        cost, delay, reliability, res_cost = self.graph.calculate_path_metrics(best_path)
        return best_path, cost, {'Cost': cost, 'Delay': delay, 'Reliability': reliability, 'Resource': res_cost}

# --- MAIN UYGULAMA (İNTERAKTİF ÇALIŞTIRMA VE DETAYLI ÇIKTI) ---
def main():
    print("--- BSM307 QoS Odaklı VNS Rotalama Başlatılıyor ---")
    
    network = NetworkGraph()
    
    try:
        network.load_data(
           'C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_NodeData.csv', 
            'C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_EdgeData.csv'
        )
    except FileNotFoundError as e:
        print(f"\nHATA: {e}")
        return

    vns = VNS_Optimizer(network)
    
    # Kullanıcıdan giriş alma
    while True:
        try:
            print("-" * 50)
            src = int(input("Başlangıç (Source) düğümünü girin: "))
            dst = int(input("Hedef (Destination) düğümünü girin: "))
            
            if src not in network.nodes or dst not in network.nodes:
                 print(f"HATA: Girdiğiniz düğümler ağda bulunmuyor. (Geçerli aralık 0 - {len(network.nodes)-1})")
            elif src == dst:
                 print("HATA: Başlangıç ve Hedef aynı olamaz.")
            else:
                break
        except ValueError:
            print("HATA: Lütfen sadece sayısal bir düğüm ID'si girin.")
            
    num_runs = 5 
    best_of_five_cost = float('inf')
    best_of_five_path = None
    best_of_five_metrics = None
    all_runs_results = [] # Tüm deneme sonuçlarını saklamak için

    
    print(f"\n--- VNS Algoritması {src} -> {dst} İçin {num_runs} Kez Çalıştırılıyor ---")
    print(f"{'Deneme':<6} | {'Maliyet':<10} | {'Gecikme':<8} | {'Güven.':<8} | {'Yol Uzunluğu':<12} | {'Yol'}")
    print("-" * 110)

    # 5 kez çalıştırma döngüsü
    for run_idx in range(num_runs):
        start_time = time.time()
        path, cost, metrics = vns.run(src, dst)
        duration = (time.time() - start_time) * 1000

        if path:
            # Deneme sonuçlarını tabloya ekle
            path_str = f"[{path[0]}...{path[-1]}]" 
            all_runs_results.append({
                'Run': run_idx + 1,
                'Cost': cost,
                'Delay': metrics['Delay'],
                'Reliability': metrics['Reliability'],
                'Length': len(path),
                'Path': path,
                'Duration': duration
            })
            
            print(f"{run_idx + 1:<6} | {cost:.4f}     | {metrics['Delay']:.1f}ms   | {metrics['Reliability']:.4f}   | {len(path):<12} | {path}")

            if cost < best_of_five_cost:
                best_of_five_cost = cost
                best_of_five_path = path
                best_of_five_metrics = metrics
        else:
            print(f"{run_idx + 1:<6} | {'-':<10} | {'-':<8} | {'-':<8} | {'-':<12} | ❌ Yol bulunamadı.")


    # Sonuçları göster
    print("\n" + "=" * 110)
    if best_of_five_path:
        
        cost = best_of_five_metrics['Cost']
        delay = best_of_five_metrics['Delay']
        reliability = best_of_five_metrics['Reliability']
        
        # Yolun ilk 3 ve son 3 düğümünü göster
        path_str = f"[{best_of_five_path[0]}, {best_of_five_path[1]}, {best_of_five_path[2]} ... {best_of_five_path[-3]}, {best_of_five_path[-2]}, {best_of_five_path[-1]}] ({len(best_of_five_path)} adım)" if len(best_of_five_path) > 6 else str(best_of_five_path)
        
        print(f"⭐ EN İYİ BULUNAN YOL ({num_runs} Deneme Sonucu):")
        print(f"   Kaynak -> Hedef: {src} -> {dst}")
        print(f"   Toplam Maliyet (Minimize Edilen): {cost:.4f}")
        print(f"   Gecikme (ms): {delay:.1f}")
        print(f"   Güvenilirlik (Maksimize Edilen): {reliability:.4f}")
        print(f"   Tam Yol (Tüm Düğümler): {best_of_five_path}")
        print(f"   Kısaltılmış Gösterim: {path_str}")

        # Tüm denemelerin performans özeti
        avg_cost = sum(r['Cost'] for r in all_runs_results) / len(all_runs_results) if all_runs_results else 0
        print("\n--- Tüm Denemelerin Performans Özeti ---")
        print(f"   Ortalama Maliyet: {avg_cost:.4f}")
        print(f"   Deneme Sayısı: {len(all_runs_results)} / {num_runs}")

    else:
        print("❌ HATA: 5 denemenin hiçbirinde geçerli bir yol bulunamadı.")
    print("=" * 110)

if __name__ == "__main__":
    main()