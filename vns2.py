import networkx as nx
import random
import math
import copy
import pandas as pd
import numpy as np
import time
from collections import Counter

# =========================================================
# GLOBAL RANDOM SEED (HER ÇALIŞTIRMADA FARKLI)
# =========================================================
random.seed(time.time())
np.random.seed(int(time.time() * 1000) % 10000)

# ---------------------------------------------------------
# 1. VERİ OKUMA VE ÖN İŞLEME
# ---------------------------------------------------------

def read_csv_robust(file_name, sep=','):
    """CSV dosyasını okur ve ön işler."""
    df = pd.read_csv(file_name, sep=sep)
    df.columns = df.columns.str.strip() 

    for col in df.columns:
        if df[col].dtype == 'object' and df[col].astype(str).str.contains(',').any():
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
    return df


# ---------------------------------------------------------
# 2. BSM307 VNS SINIFI
# ---------------------------------------------------------

class BSM307VNS:
    def __init__(self, node_file, edge_file):
        # HER ÇALIŞTIRMADA FARKLI AĞIRLIKLAR (Toplamı 1.0)
        w = np.random.rand(3)
        w = w / w.sum()

        self.w_delay = w[0]
        self.w_reliability = w[1] 
        self.w_resource = w[2]

        self.graph = nx.DiGraph()
        self.load_network(node_file, edge_file)

    def load_network(self, node_file, edge_file):
        node_df = read_csv_robust(node_file)
        edge_df = read_csv_robust(edge_file)

        for _, row in node_df.iterrows():
            self.graph.add_node(
                int(row['node_id']),
                processing_delay=row['s_ms'],
                reliability=row['r_node']
            )

        for _, row in edge_df.iterrows():
            u, v = int(row['src']), int(row['dst'])
            
            link_attrs = {
                'capacity_mbps': row['capacity_mbps'],
                'delay_ms': row['delay_ms'],
                'reliability': row['r_link']
            }

            self.graph.add_edge(u, v, **link_attrs)
            
            if not self.graph.has_edge(v, u):
                self.graph.add_edge(v, u, **link_attrs)


    def _check_max_node_visits(self, path, max_visits=5):
        """Bir yoldaki her düğümün ziyaret sayısını kontrol eder."""
        if not path:
            return False
        
        node_counts = Counter(path)
        for node, count in node_counts.items():
            if count > max_visits:
                return False
        return True

    def calculate_metrics(self, path, demand):
        """Yolun metriklerini hesaplar ve tek amaçlı maliyet fonksiyonunu döndürür."""
        if not path or len(path) < 2:
            return float('inf'), 0, 0, 0, False

        if not self._check_max_node_visits(path, max_visits=5):
            return float('inf'), 0, 0, 0, False

        total_delay = 0
        log_rel = 0
        resource = 0
        real_rel = 1.0

        for node in path[1:-1]:
            if node in self.graph.nodes:
                n = self.graph.nodes[node]
                total_delay += n['processing_delay']
                log_rel += -math.log(n['reliability'])
                real_rel *= n['reliability']

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            if not self.graph.has_edge(u, v):
                return float('inf'), 0, 0, 0, False

            e = self.graph.edges[u, v]

            if e['capacity_mbps'] < demand:
                return float('inf'), 0, 0, 0, False

            total_delay += e['delay_ms']
            log_rel += -math.log(e['reliability'])
            resource += 1000.0 / e['capacity_mbps']
            real_rel *= e['reliability']

        # TEK AMAÇLI MALİYET FONKSİYONU
        cost = (
            self.w_delay * total_delay +
            self.w_reliability * log_rel + 
            self.w_resource * resource
        )

        return cost, total_delay, real_rel, resource, True

    def get_initial_solution(self, source, target, demand):
        """Kapasite kısıtlamasına uyan kenarlar üzerinde en kısa yolu bulur."""
        valid_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if d['capacity_mbps'] >= demand
        ]
        temp_graph = nx.DiGraph(valid_edges)

        try:
            return nx.shortest_path(temp_graph, source, target, weight='delay_ms')
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None

    def shaking(self, path, demand):
        """Yolun rastgele bir bölümünü keser ve yerine alternatif, kısa bir yol bulur."""
        new_path = copy.deepcopy(path)
        if len(new_path) < 4:
            return new_path

        i = random.randint(0, len(new_path) - 3)
        j = random.randint(i + 2, len(new_path) - 1)

        start_node = new_path[i]
        end_node = new_path[j]

        valid_edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if d['capacity_mbps'] >= demand
        ]
        temp_graph = nx.DiGraph(valid_edges)

        try:
            path_to_insert = nx.shortest_path(temp_graph, start_node, end_node, weight='delay_ms')
            
            return new_path[:i] + path_to_insert + new_path[j + 1:]
            
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return new_path

    def local_search(self, path, demand):
        """Yerel iyileştirme adımı: Tek bir düğümü atlamayı dener."""
        best_path = list(path)
        best_cost, *_ , feasible = self.calculate_metrics(best_path, demand)
        
        if not feasible:
            return path
        
        improved = True
        while improved:
            improved = False
            for idx in range(1, len(best_path) - 1):
                u, v = best_path[idx-1], best_path[idx+1]
                
                if self.graph.has_edge(u, v):
                    e = self.graph.edges[u, v]
                    
                    if e['capacity_mbps'] >= demand:
                        candidate = best_path[:idx] + best_path[idx+1:]
                        cost, *_ , feasible = self.calculate_metrics(candidate, demand)

                        if feasible and cost < best_cost:
                            best_path = candidate
                            best_cost = cost
                            improved = True
                            break
        
        return best_path

    def run_vns(self, source, target, demand, max_attempts=50):
        
        path = self.get_initial_solution(source, target, demand)
        cost, _, _, _, feasible = self.calculate_metrics(path, demand)
        
        if not feasible:
            return None, float('inf'), 0, 0, 0, 0

        best_path = path
        best_cost = cost
        current_path = path

        for _ in range(max_attempts):
            k = 1
            k_max = 4 

            while k <= k_max:
                shaken = self.shaking(current_path, demand)
                improved = self.local_search(shaken, demand)
                cost, _, _, _, feasible = self.calculate_metrics(improved, demand)

                if feasible and cost < best_cost:
                    best_path = improved
                    best_cost = cost
                    current_path = improved 
                    k = 1
                else:
                    k += 1

            current_path = best_path
        
        cost, delay, rel, res, _ = self.calculate_metrics(best_path, demand)
        return best_path, cost, delay, rel, res, len(best_path)


# ---------------------------------------------------------
# 3. ANA PROGRAM (SADELEŞTİRİLMİŞ ÇIKTI)
# ---------------------------------------------------------

if __name__ == "__main__":

    NODE_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_NodeData.csv"
    EDGE_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_EdgeData.csv"
    DEMAND_FILE = "C:\\Users\\yigit\\OneDrive\\Masaüstü\Vns_Algorithm\\Vns_Algorithm\\BSM307_317_Guz2025_TermProject_DemandData.csv"

    demand_df = read_csv_robust(DEMAND_FILE)
    all_results = []

    print("\n" + "="*80)
    print("BSM307 TEK AMAÇLI VNS ALGORİTMASI")
    print("="*80)
    
    print("Tüm talepler 5 farklı rastgele amaç ağırlığı ile test ediliyor...")
    
    for index, row in demand_df.iterrows():
        S, D, demand = int(row['src']), int(row['dst']), row['demand_mbps']

        current_demand_results = []
        
        # Her talep için 5 deneme yapılır
        for run_id in range(1, 6):
            
            # 🔥 Her tekrarda yeni, rastgele ağırlıklı motor oluşturulur
            vns_engine = BSM307VNS(NODE_FILE, EDGE_FILE) 
            current_w_info = (f"D={vns_engine.w_delay:.2f} R={vns_engine.w_reliability:.2f} Res={vns_engine.w_resource:.2f}")

            path, cost, delay, rel, res, steps = vns_engine.run_vns(S, D, demand)
            
            current_demand_results.append({
                'run_id': run_id,
                'path': path,
                'cost': cost,
                'delay': delay,
                'reliability': rel,
                'resource_usage': res,
                'steps': steps,
                'weights': current_w_info
            })
            
        # Her talep için en iyi sonucu bul
        best_run = min(current_demand_results, key=lambda x: x['cost'])
        all_results.append({
            'demand_id': index + 1,
            'source': S,
            'target': D,
            'demand_mbps': demand,
            'best_run': best_run
        })

    # --- GENEL SONUÇ TABLOSU ---
    
    print("\n" + "="*120)
    print("| GENEL BAŞARI ÖZETİ (Tüm Taleplerin 5 Deneme Arasındaki En İyi Çözümü) |")
    print("="*120)
    
    # Yeni başlık, en temel metrikleri ve ağırlık bilgisini içerir.
    header = "{:<5} | {:<5}->{:<5} | {:<7} | {:<12} | {:<11} | {:<40}"
    # Başlık sırası: Talep(ID), Kaynak, Hedef, Maliyet, Gecikme(ms), Güvenilirlik, Kullanılan Ağırlıklar
    print(header.format("Talep", "Src", "Dst", "Maliyet", "Gecikme(ms)", "Güvenilirlik", "Ağırlıklar (D/R/Res)"))
    print("-" * 120)

    for result in all_results:
        best = result['best_run']
        
        if best['path'] and best['cost'] != float('inf'):
            rel_percent = best['reliability'] * 100
            
            # 1. SATIR: Temel Metrikler ve Ağırlıklar
            print(header.format(
                result['demand_id'],
                result['source'],
                result['target'],
                f"{best['cost']:.4f}",
                f"{best['delay']:.2f}",
                f"%{rel_percent:.2f}",
                best['weights']
            ))
            
            # 2. SATIR: Kaynak Kullanımı ve Yol Özeti
            path_str = ' -> '.join(map(str, best['path'][:3]))
            steps = best['steps']
            if steps > 6:
                path_str += f" -> ... ({steps-1} kenar) -> " + ' -> '.join(map(str, best['path'][-3:]))
            else:
                path_str = ' -> '.join(map(str, best['path']))
            
            # Kaynak Kullanımı ve Yol bilgisi daha kısa bir girintiyle gösterilir
            print(f"{' '*20} Kullanım: {best['resource_usage']:.2f} | Yol: {path_str}")
        else:
            # Başarısız sonuç
            print(f"{result['demand_id']:<5} | {result['source']:<5}->{result['target']:<5} | **UYGUN YOL BULUNAMADI**")
            
    print("=" * 120)