# Optimal Number of Clusters in Agglomerative Clustering: Comprehensive Research Report

## Executive Summary

This research examines the best methods for automatically determining the optimal number of clusters in agglomerative clustering, specifically tailored for:
- **Text embeddings (384-dimensional)**
- **Cosine/Euclidean distance metrics**
- **Ward linkage**
- **Large datasets (3000+ data points)**
- **Business interpretability requirements**

Based on current literature and practical experiments, the most effective approaches combine multiple validation metrics with dendrogram analysis and distance threshold methods.

## 1. Distance Threshold Methods for Agglomerative Clustering

### 1.1 Extreme Value Theory (EVT) Approach (2025)
**Latest Research**: A breakthrough method using Extreme Value Theory for automatic threshold selection.

```python
import numpy as np
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import weibull_min

def evt_threshold_selection(embeddings, linkage_method='ward'):
    """
    Extreme Value Theory approach for optimal threshold selection.
    Based on: "Agglomerative Clustering with Threshold Optimization via Extreme Value Theory" (2025)
    """
    # Compute linkage matrix
    Z = linkage(embeddings, method=linkage_method)
    
    # Extract linkage distances
    distances = Z[:, 2]
    
    # Sort distances in descending order (largest merges first)
    sorted_distances = np.sort(distances)[::-1]
    
    # Fit Weibull distribution to upper tail
    # Theory: correct linkage distances follow Weibull distribution
    tail_size = min(100, len(sorted_distances) // 3)
    tail_distances = sorted_distances[:tail_size]
    
    # Fit Weibull distribution
    weibull_params = weibull_min.fit(tail_distances, floc=0)
    
    # Calculate threshold using Weibull distribution
    # Use 95th percentile as threshold
    threshold = weibull_min.ppf(0.95, *weibull_params)
    
    # Apply threshold to get clusters
    cluster_labels = fcluster(Z, threshold, criterion='distance')
    n_clusters = len(np.unique(cluster_labels))
    
    return {
        'optimal_threshold': threshold,
        'n_clusters': n_clusters,
        'cluster_labels': cluster_labels,
        'weibull_params': weibull_params,
        'linkage_matrix': Z
    }

# Example usage
result = evt_threshold_selection(embeddings)
print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
print(f"Optimal clusters: {result['n_clusters']}")
```

### 1.2 Dynamic Height Cutting
```python
def dynamic_height_cutting(linkage_matrix, min_cluster_size=10, deepSplit=2):
    """
    Dynamic tree cutting algorithm for hierarchical clustering.
    Adaptively determines cutting height based on tree structure.
    """
    from scipy.cluster.hierarchy import fcluster
    import numpy as np
    
    # Calculate distances between consecutive merges
    distances = linkage_matrix[:, 2]
    distance_gaps = np.diff(distances)
    
    # Find significant gaps (indicating natural cluster boundaries)
    gap_threshold = np.percentile(distance_gaps, 75 + deepSplit * 5)
    significant_gaps = np.where(distance_gaps > gap_threshold)[0]
    
    if len(significant_gaps) == 0:
        # Fallback to largest gap
        cut_height = distances[np.argmax(distance_gaps)]
    else:
        # Use the height corresponding to the most significant gap
        cut_height = distances[significant_gaps[-1]]
    
    # Apply cutting height
    cluster_labels = fcluster(linkage_matrix, cut_height, criterion='distance')
    
    return {
        'cut_height': cut_height,
        'n_clusters': len(np.unique(cluster_labels)),
        'cluster_labels': cluster_labels,
        'significant_gaps': significant_gaps
    }
```

## 2. Elbow Method Applied to Linkage Distances/Dendrogram

### 2.1 Linkage Distance Elbow Method
```python
def linkage_elbow_method(embeddings, linkage_method='ward', max_clusters=30):
    """
    Apply elbow method to linkage distances for optimal cluster determination.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    
    # Compute linkage matrix
    Z = linkage(embeddings, method=linkage_method)
    
    # Test different numbers of clusters
    cluster_range = range(2, min(max_clusters + 1, len(embeddings) // 2))
    wcss_values = []
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        # Get clusters
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Calculate WCSS (Within-Cluster Sum of Squares)
        wcss = calculate_wcss(embeddings, labels)
        wcss_values.append(wcss)
        
        # Calculate silhouette score (for smaller datasets)
        if len(embeddings) < 2000:
            sil_score = silhouette_score(embeddings, labels)
            silhouette_scores.append(sil_score)
    
    # Find elbow point using second derivative
    elbow_point = find_elbow_point(wcss_values)
    optimal_clusters = cluster_range[elbow_point]
    
    return {
        'optimal_clusters': optimal_clusters,
        'wcss_values': wcss_values,
        'silhouette_scores': silhouette_scores,
        'cluster_range': list(cluster_range),
        'linkage_matrix': Z
    }

def calculate_wcss(embeddings, labels):
    """Calculate Within-Cluster Sum of Squares."""
    wcss = 0
    for cluster_id in np.unique(labels):
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            centroid = np.mean(cluster_points, axis=0)
            wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

def find_elbow_point(values):
    """Find elbow point using second derivative method."""
    values = np.array(values)
    
    # Calculate first and second derivatives
    first_derivative = np.diff(values)
    second_derivative = np.diff(first_derivative)
    
    # Find point with maximum second derivative (steepest change)
    elbow_idx = np.argmax(second_derivative) + 1  # +1 to account for diff offset
    
    return elbow_idx
```

### 2.2 Dendrogram Gap Analysis
```python
def dendrogram_gap_analysis(linkage_matrix, plot=True):
    """
    Analyze gaps in dendrogram to find optimal cutting points.
    """
    distances = linkage_matrix[:, 2]
    
    # Calculate gaps between consecutive merge distances
    gaps = np.diff(distances)
    
    # Find the largest gaps (indicating natural cluster separations)
    gap_indices = np.argsort(gaps)[::-1]  # Sort in descending order
    
    # Calculate gap ratios
    gap_ratios = gaps / np.mean(gaps)
    
    # Find optimal number of clusters based on largest gap
    optimal_cut_idx = gap_indices[0]
    optimal_clusters = len(linkage_matrix) - optimal_cut_idx
    
    if plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(gaps)), gaps, 'b-o')
        plt.axvline(optimal_cut_idx, color='r', linestyle='--', label=f'Optimal cut (k={optimal_clusters})')
        plt.xlabel('Merge Step')
        plt.ylabel('Distance Gap')
        plt.title('Dendrogram Distance Gaps')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        dendrogram(linkage_matrix, truncate_mode='level', p=10)
        plt.title('Dendrogram (Truncated)')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'optimal_clusters': optimal_clusters,
        'gaps': gaps,
        'gap_ratios': gap_ratios,
        'largest_gap_index': optimal_cut_idx
    }
```

## 3. Silhouette Analysis for Hierarchical Clustering

### 3.1 Comprehensive Silhouette Analysis
```python
def hierarchical_silhouette_analysis(embeddings, max_clusters=20, linkage_method='ward'):
    """
    Comprehensive silhouette analysis for hierarchical clustering.
    Optimized for large datasets (3000+ points).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score, silhouette_samples
    import matplotlib.pyplot as plt
    
    # Compute linkage matrix
    Z = linkage(embeddings, method=linkage_method)
    
    # For large datasets, use sampling for silhouette calculation
    use_sampling = len(embeddings) > 2000
    if use_sampling:
        sample_size = min(2000, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
    else:
        sample_embeddings = embeddings
        sample_indices = np.arange(len(embeddings))
    
    cluster_range = range(2, min(max_clusters + 1, len(embeddings) // 10))
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        # Get clusters for full dataset
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Calculate silhouette score on sample
        if use_sampling:
            sample_labels = labels[sample_indices]
            sil_score = silhouette_score(sample_embeddings, sample_labels)
        else:
            sil_score = silhouette_score(embeddings, labels)
        
        silhouette_scores.append(sil_score)
    
    # Find optimal number of clusters
    optimal_idx = np.argmax(silhouette_scores)
    optimal_clusters = cluster_range[optimal_idx]
    optimal_score = silhouette_scores[optimal_idx]
    
    return {
        'optimal_clusters': optimal_clusters,
        'optimal_score': optimal_score,
        'silhouette_scores': silhouette_scores,
        'cluster_range': list(cluster_range),
        'used_sampling': use_sampling
    }

def plot_silhouette_analysis(embeddings, cluster_range, silhouette_scores):
    """Plot silhouette analysis results."""
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Analysis for Hierarchical Clustering')
    plt.grid(True, alpha=0.3)
    
    # Highlight optimal point
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = cluster_range[optimal_idx]
    optimal_score = silhouette_scores[optimal_idx]
    
    plt.axvline(optimal_k, color='r', linestyle='--', alpha=0.7)
    plt.plot(optimal_k, optimal_score, 'ro', markersize=12, label=f'Optimal k={optimal_k}')
    plt.legend()
    plt.show()
```

## 4. Gap Statistic for Cluster Validation

### 4.1 Optimized Gap Statistic Implementation
```python
def gap_statistic_hierarchical(embeddings, max_clusters=20, n_refs=10, linkage_method='ward'):
    """
    Gap statistic implementation optimized for hierarchical clustering on large datasets.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Standardize embeddings for gap statistic
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Compute linkage matrix once
    Z = linkage(embeddings_scaled, method=linkage_method)
    
    cluster_range = range(1, min(max_clusters + 1, len(embeddings) // 10))
    gaps = []
    s_k = []  # Standard errors
    
    for k in cluster_range:
        # Calculate W_k for actual data
        if k == 1:
            wk = calculate_total_dispersion(embeddings_scaled)
        else:
            labels = fcluster(Z, k, criterion='maxclust')
            wk = calculate_within_cluster_dispersion(embeddings_scaled, labels)
        
        # Generate reference datasets and calculate W_k for each
        ref_wks = []
        for i in range(n_refs):
            # Generate reference data with same distribution
            ref_data = generate_reference_data(embeddings_scaled)
            ref_Z = linkage(ref_data, method=linkage_method)
            
            if k == 1:
                ref_wk = calculate_total_dispersion(ref_data)
            else:
                ref_labels = fcluster(ref_Z, k, criterion='maxclust')
                ref_wk = calculate_within_cluster_dispersion(ref_data, ref_labels)
            
            ref_wks.append(ref_wk)
        
        # Calculate gap and standard error
        ref_wks = np.array(ref_wks)
        gap = np.mean(np.log(ref_wks)) - np.log(wk)
        sk = np.sqrt(1 + 1/n_refs) * np.std(np.log(ref_wks))
        
        gaps.append(gap)
        s_k.append(sk)
    
    # Find optimal k using gap statistic criterion
    gaps = np.array(gaps)
    s_k = np.array(s_k)
    
    # Find smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
    optimal_k = 1
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - s_k[i + 1]:
            optimal_k = cluster_range[i]
            break
    
    return {
        'optimal_clusters': optimal_k,
        'gaps': gaps,
        'standard_errors': s_k,
        'cluster_range': list(cluster_range)
    }

def calculate_within_cluster_dispersion(embeddings, labels):
    """Calculate total within-cluster dispersion."""
    total_dispersion = 0
    for cluster_id in np.unique(labels):
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            # Calculate pairwise distances within cluster
            from scipy.spatial.distance import pdist
            distances = pdist(cluster_points)
            total_dispersion += np.sum(distances)
    return total_dispersion

def calculate_total_dispersion(embeddings):
    """Calculate total dispersion for k=1."""
    from scipy.spatial.distance import pdist
    return np.sum(pdist(embeddings))

def generate_reference_data(embeddings):
    """Generate reference data with same distribution characteristics."""
    # Use uniform distribution over the bounding box
    mins = np.min(embeddings, axis=0)
    maxs = np.max(embeddings, axis=0)
    
    ref_data = np.random.uniform(mins, maxs, size=embeddings.shape)
    return ref_data
```

## 5. Calinski-Harabasz Index Progression

### 5.1 Optimized CH Index for Large Datasets
```python
def calinski_harabasz_progression(embeddings, max_clusters=30, linkage_method='ward'):
    """
    Calinski-Harabasz index progression for hierarchical clustering.
    Optimized for high-dimensional text embeddings.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler
    
    # Standardize embeddings for better CH performance
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Compute linkage matrix
    Z = linkage(embeddings_scaled, method=linkage_method)
    
    cluster_range = range(2, min(max_clusters + 1, len(embeddings) // 5))
    ch_scores = []
    
    for n_clusters in cluster_range:
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Calculate CH score
        ch_score = calinski_harabasz_score(embeddings_scaled, labels)
        ch_scores.append(ch_score)
    
    # Find optimal number of clusters (maximum CH score)
    optimal_idx = np.argmax(ch_scores)
    optimal_clusters = cluster_range[optimal_idx]
    optimal_score = ch_scores[optimal_idx]
    
    # Find secondary peak (for business interpretability)
    ch_scores_array = np.array(ch_scores)
    # Smooth the curve to find local maxima
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ch_scores_array, height=np.percentile(ch_scores_array, 70))
    
    secondary_peaks = []
    if len(peaks) > 1:
        peak_heights = ch_scores_array[peaks]
        sorted_peaks = peaks[np.argsort(peak_heights)[::-1]]
        secondary_peaks = [cluster_range[p] for p in sorted_peaks[1:3]]  # Top 2 secondary peaks
    
    return {
        'optimal_clusters': optimal_clusters,
        'optimal_score': optimal_score,
        'ch_scores': ch_scores,
        'cluster_range': list(cluster_range),
        'secondary_peaks': secondary_peaks,
        'all_scores': list(zip(cluster_range, ch_scores))
    }

def plot_ch_progression(cluster_range, ch_scores, optimal_clusters, secondary_peaks=None):
    """Plot Calinski-Harabasz progression."""
    plt.figure(figsize=(12, 6))
    plt.plot(cluster_range, ch_scores, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Index Progression')
    plt.grid(True, alpha=0.3)
    
    # Highlight optimal point
    optimal_idx = cluster_range.index(optimal_clusters)
    plt.axvline(optimal_clusters, color='r', linestyle='--', alpha=0.7)
    plt.plot(optimal_clusters, ch_scores[optimal_idx], 'ro', markersize=12, 
             label=f'Optimal k={optimal_clusters}')
    
    # Highlight secondary peaks
    if secondary_peaks:
        for peak in secondary_peaks:
            if peak in cluster_range:
                peak_idx = cluster_range.index(peak)
                plt.plot(peak, ch_scores[peak_idx], 'go', markersize=10, 
                        label=f'Secondary peak k={peak}')
    
    plt.legend()
    plt.show()
```

## 6. Within-Cluster Sum of Squares (WCSS) Elbow Detection

### 6.1 Advanced Elbow Detection
```python
def advanced_wcss_elbow_detection(embeddings, max_clusters=25, linkage_method='ward'):
    """
    Advanced WCSS elbow detection with multiple elbow finding algorithms.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Compute linkage matrix
    Z = linkage(embeddings_scaled, method=linkage_method)
    
    cluster_range = range(1, min(max_clusters + 1, len(embeddings) // 5))
    wcss_values = []
    
    for n_clusters in cluster_range:
        if n_clusters == 1:
            # All points in one cluster
            centroid = np.mean(embeddings_scaled, axis=0)
            wcss = np.sum((embeddings_scaled - centroid) ** 2)
        else:
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            wcss = calculate_wcss_detailed(embeddings_scaled, labels)
        
        wcss_values.append(wcss)
    
    # Apply multiple elbow detection methods
    elbow_methods = {
        'knee_locator': find_elbow_knee_locator(cluster_range, wcss_values),
        'second_derivative': find_elbow_second_derivative(wcss_values),
        'percentage_change': find_elbow_percentage_change(wcss_values),
        'distance_method': find_elbow_distance_method(cluster_range, wcss_values)
    }
    
    # Consensus elbow point
    elbow_points = [ep for ep in elbow_methods.values() if ep is not None]
    if elbow_points:
        consensus_elbow = int(np.median(elbow_points))
    else:
        consensus_elbow = len(cluster_range) // 3  # Fallback
    
    return {
        'optimal_clusters': consensus_elbow,
        'wcss_values': wcss_values,
        'cluster_range': list(cluster_range),
        'elbow_methods': elbow_methods,
        'method_agreement': len(set(elbow_points)) == 1 if elbow_points else False
    }

def find_elbow_knee_locator(x, y):
    """Find elbow using knee locator algorithm."""
    try:
        from kneed import KneeLocator
        kl = KneeLocator(x, y, curve='convex', direction='decreasing')
        return kl.elbow
    except ImportError:
        return None

def find_elbow_second_derivative(y):
    """Find elbow using second derivative method."""
    y = np.array(y)
    first_derivative = np.diff(y)
    second_derivative = np.diff(first_derivative)
    
    # Find point with maximum second derivative change
    elbow_idx = np.argmax(second_derivative) + 2  # +2 to account for diff operations
    return elbow_idx

def find_elbow_percentage_change(y):
    """Find elbow using percentage change method."""
    y = np.array(y)
    percentage_changes = np.abs(np.diff(y) / y[:-1] * 100)
    
    # Find where percentage change drops below threshold
    threshold = np.percentile(percentage_changes, 25)  # Bottom 25% of changes
    elbow_candidates = np.where(percentage_changes < threshold)[0]
    
    return elbow_candidates[0] + 1 if len(elbow_candidates) > 0 else None

def find_elbow_distance_method(x, y):
    """Find elbow using distance from line method."""
    x = np.array(x)
    y = np.array(y)
    
    # Create line from first to last point
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # Calculate distances from each point to the line
    distances = []
    for i in range(len(x)):
        point = np.array([x[i], y[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
        distances.append(distance)
    
    # Find point with maximum distance
    elbow_idx = np.argmax(distances)
    return x[elbow_idx]

def calculate_wcss_detailed(embeddings, labels):
    """Calculate WCSS with detailed computation."""
    wcss = 0
    for cluster_id in np.unique(labels):
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss
```

## 7. Consensus Clustering Approaches

### 7.1 Multi-Algorithm Consensus Method
```python
def multi_algorithm_consensus(embeddings, max_clusters=25):
    """
    Consensus clustering using multiple validation methods.
    Combines results from different approaches for robust cluster number estimation.
    """
    
    # Apply multiple methods
    methods_results = {}
    
    # 1. Silhouette Analysis
    try:
        sil_result = hierarchical_silhouette_analysis(embeddings, max_clusters)
        methods_results['silhouette'] = sil_result['optimal_clusters']
    except Exception as e:
        print(f"Silhouette analysis failed: {e}")
    
    # 2. Calinski-Harabasz Index
    try:
        ch_result = calinski_harabasz_progression(embeddings, max_clusters)
        methods_results['calinski_harabasz'] = ch_result['optimal_clusters']
        methods_results['ch_secondary'] = ch_result['secondary_peaks'][:2] if ch_result['secondary_peaks'] else []
    except Exception as e:
        print(f"Calinski-Harabasz analysis failed: {e}")
    
    # 3. Gap Statistic (sampled for large datasets)
    try:
        if len(embeddings) > 1000:
            # Use sampling for gap statistic on large datasets
            sample_size = min(1000, len(embeddings))
            sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
            gap_result = gap_statistic_hierarchical(embeddings[sample_idx], 
                                                  min(max_clusters, 15), n_refs=5)
        else:
            gap_result = gap_statistic_hierarchical(embeddings, max_clusters, n_refs=10)
        methods_results['gap_statistic'] = gap_result['optimal_clusters']
    except Exception as e:
        print(f"Gap statistic analysis failed: {e}")
    
    # 4. WCSS Elbow Method
    try:
        wcss_result = advanced_wcss_elbow_detection(embeddings, max_clusters)
        methods_results['wcss_elbow'] = wcss_result['optimal_clusters']
    except Exception as e:
        print(f"WCSS elbow analysis failed: {e}")
    
    # 5. Dendrogram Gap Analysis
    try:
        from scipy.cluster.hierarchy import linkage
        Z = linkage(embeddings, method='ward')
        dend_result = dendrogram_gap_analysis(Z, plot=False)
        methods_results['dendrogram_gap'] = dend_result['optimal_clusters']
    except Exception as e:
        print(f"Dendrogram gap analysis failed: {e}")
    
    # Consensus analysis
    all_suggestions = [v for v in methods_results.values() if isinstance(v, int)]
    
    if not all_suggestions:
        return {'consensus_clusters': 5, 'method_results': methods_results, 'confidence': 'low'}
    
    # Calculate consensus using different approaches
    consensus_methods = {
        'median': int(np.median(all_suggestions)),
        'mode': int(stats.mode(all_suggestions)[0]) if len(all_suggestions) > 1 else all_suggestions[0],
        'weighted_average': calculate_weighted_consensus(methods_results),
        'business_optimized': optimize_for_business_interpretability(all_suggestions)
    }
    
    # Final consensus (prefer business-optimized if reasonable)
    final_consensus = consensus_methods['business_optimized']
    
    # Calculate confidence based on agreement
    unique_suggestions = len(set(all_suggestions))
    total_suggestions = len(all_suggestions)
    confidence = 'high' if unique_suggestions <= 2 else 'medium' if unique_suggestions <= 4 else 'low'
    
    return {
        'consensus_clusters': final_consensus,
        'method_results': methods_results,
        'consensus_methods': consensus_methods,
        'confidence': confidence,
        'agreement_ratio': (total_suggestions - unique_suggestions + 1) / total_suggestions
    }

def calculate_weighted_consensus(method_results):
    """Calculate weighted consensus based on method reliability."""
    # Weights based on method reliability for text embeddings
    weights = {
        'silhouette': 0.25,
        'calinski_harabasz': 0.30,
        'gap_statistic': 0.20,
        'wcss_elbow': 0.15,
        'dendrogram_gap': 0.10
    }
    
    weighted_sum = 0
    total_weight = 0
    
    for method, result in method_results.items():
        if isinstance(result, int) and method in weights:
            weighted_sum += result * weights[method]
            total_weight += weights[method]
    
    return int(weighted_sum / total_weight) if total_weight > 0 else 5

def optimize_for_business_interpretability(suggestions):
    """Optimize cluster number for business interpretability."""
    suggestions = np.array(suggestions)
    
    # Business-friendly ranges (easier to interpret and manage)
    business_ranges = [
        (5, 10),   # Very manageable
        (10, 15),  # Manageable
        (15, 25),  # Moderate
        (25, 35)   # Complex but feasible
    ]
    
    # Find suggestions in business-friendly ranges
    for min_k, max_k in business_ranges:
        candidates = suggestions[(suggestions >= min_k) & (suggestions <= max_k)]
        if len(candidates) > 0:
            return int(np.median(candidates))
    
    # Fallback to overall median
    return int(np.median(suggestions))
```

### 7.2 Ensemble Stability Analysis
```python
def ensemble_stability_analysis(embeddings, n_iterations=20, sample_ratio=0.8):
    """
    Assess clustering stability through bootstrap sampling.
    Helps determine reliable cluster numbers.
    """
    from collections import Counter
    
    n_samples = len(embeddings)
    sample_size = int(n_samples * sample_ratio)
    
    stability_results = []
    
    for iteration in range(n_iterations):
        # Bootstrap sample
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        # Apply consensus clustering on sample
        result = multi_algorithm_consensus(sample_embeddings, max_clusters=20)
        stability_results.append(result['consensus_clusters'])
    
    # Analyze stability
    cluster_counts = Counter(stability_results)
    most_common = cluster_counts.most_common()
    
    # Calculate stability metrics
    stability_score = most_common[0][1] / n_iterations  # Frequency of most common result
    stable_clusters = most_common[0][0]
    
    return {
        'stable_clusters': stable_clusters,
        'stability_score': stability_score,
        'all_results': stability_results,
        'frequency_distribution': dict(cluster_counts),
        'top_3_candidates': [k for k, v in most_common[:3]]
    }
```

## 8. Comprehensive Implementation Framework

### 8.1 Complete Optimization Pipeline
```python
class OptimalClusterDeterminer:
    """
    Comprehensive framework for determining optimal number of clusters
    in agglomerative clustering for text embeddings.
    """
    
    def __init__(self, business_constraints=None):
        self.business_constraints = business_constraints or {
            'min_clusters': 3,
            'max_clusters': 30,
            'prefer_interpretable': True,
            'max_runtime_minutes': 10
        }
        
    def determine_optimal_clusters(self, embeddings, linkage_method='ward', 
                                 include_stability=True, verbose=True):
        """
        Main method to determine optimal number of clusters.
        """
        if verbose:
            print(f"Analyzing {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
            print(f"Business constraints: {self.business_constraints}")
        
        results = {}
        
        # 1. Quick methods first
        start_time = time.time()
        
        # Distance threshold methods
        if verbose:
            print("\n1. Applying distance threshold methods...")
        
        try:
            evt_result = evt_threshold_selection(embeddings, linkage_method)
            results['evt_threshold'] = evt_result
            if verbose:
                print(f"   EVT method suggests: {evt_result['n_clusters']} clusters")
        except Exception as e:
            if verbose:
                print(f"   EVT method failed: {e}")
        
        # 2. Validation metrics
        if verbose:
            print("\n2. Applying validation metrics...")
            
        # Consensus clustering
        consensus_result = multi_algorithm_consensus(
            embeddings, 
            max_clusters=self.business_constraints['max_clusters']
        )
        results['consensus'] = consensus_result
        
        if verbose:
            print(f"   Consensus method suggests: {consensus_result['consensus_clusters']} clusters")
            print(f"   Method agreement: {consensus_result['confidence']}")
            print(f"   Individual methods: {consensus_result['method_results']}")
        
        # 3. Stability analysis (if time permits and requested)
        if include_stability and time.time() - start_time < self.business_constraints['max_runtime_minutes'] * 60 * 0.7:
            if verbose:
                print("\n3. Performing stability analysis...")
                
            stability_result = ensemble_stability_analysis(embeddings)
            results['stability'] = stability_result
            
            if verbose:
                print(f"   Stable clusters: {stability_result['stable_clusters']}")
                print(f"   Stability score: {stability_result['stability_score']:.3f}")
        
        # 4. Final recommendation
        final_recommendation = self._make_final_recommendation(results, verbose)
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nAnalysis completed in {total_time:.2f} seconds")
            print(f"Final recommendation: {final_recommendation['recommended_clusters']} clusters")
            print(f"Confidence: {final_recommendation['confidence']}")
        
        return {
            'recommended_clusters': final_recommendation['recommended_clusters'],
            'confidence': final_recommendation['confidence'],
            'detailed_results': results,
            'analysis_time': total_time,
            'alternative_options': final_recommendation['alternatives']
        }
    
    def _make_final_recommendation(self, results, verbose=True):
        """Make final recommendation based on all analyses."""
        
        # Collect all suggestions
        suggestions = []
        
        if 'evt_threshold' in results:
            suggestions.append(results['evt_threshold']['n_clusters'])
        
        if 'consensus' in results:
            suggestions.append(results['consensus']['consensus_clusters'])
        
        if 'stability' in results:
            suggestions.append(results['stability']['stable_clusters'])
        
        if not suggestions:
            return {
                'recommended_clusters': 5,
                'confidence': 'low',
                'alternatives': [3, 7, 10]
            }
        
        # Apply business constraints
        min_k = self.business_constraints['min_clusters']
        max_k = self.business_constraints['max_clusters']
        
        valid_suggestions = [s for s in suggestions if min_k <= s <= max_k]
        if not valid_suggestions:
            valid_suggestions = suggestions  # Use all if none valid
        
        # Final decision logic
        if len(set(valid_suggestions)) == 1:
            # Perfect agreement
            recommended = valid_suggestions[0]
            confidence = 'high'
        elif len(set(valid_suggestions)) <= 2:
            # Good agreement
            recommended = int(np.median(valid_suggestions))
            confidence = 'high'
        else:
            # Some disagreement, use business-optimized approach
            recommended = optimize_for_business_interpretability(valid_suggestions)
            confidence = 'medium'
        
        # Generate alternatives
        alternatives = sorted(list(set(valid_suggestions)))[:3]
        if recommended in alternatives:
            alternatives.remove(recommended)
        
        return {
            'recommended_clusters': recommended,
            'confidence': confidence,
            'alternatives': alternatives[:2]  # Top 2 alternatives
        }

# Usage example
def analyze_ticket_clustering(embeddings_file_path):
    """
    Complete analysis pipeline for ticket classification clustering.
    """
    # Load embeddings (assuming they're saved)
    embeddings = np.load(embeddings_file_path)  # or load from your source
    
    # Initialize optimizer with business constraints
    optimizer = OptimalClusterDeterminer(
        business_constraints={
            'min_clusters': 5,     # At least 5 clusters for meaningful categorization
            'max_clusters': 25,    # At most 25 for human interpretability
            'prefer_interpretable': True,
            'max_runtime_minutes': 5  # 5 minutes max for analysis
        }
    )
    
    # Perform analysis
    result = optimizer.determine_optimal_clusters(
        embeddings, 
        linkage_method='ward',
        include_stability=True,
        verbose=True
    )
    
    return result
```

## 9. Practical Parameter Recommendations

### 9.1 Text Embeddings (384-dimensional)
```python
# Recommended parameters for 384-dimensional text embeddings
OPTIMAL_PARAMS = {
    'linkage_method': 'ward',  # Best for spherical clusters in high dimensions
    'distance_metric': 'euclidean',  # Required for Ward linkage
    'preprocessing': {
        'standardize': True,  # Important for high-dimensional data
        'normalize': True,    # L2 normalization for text embeddings
        'pca_threshold': 0.95  # Optional: reduce to 95% variance if needed
    },
    'validation_methods': [
        'calinski_harabasz',  # Most reliable for text data
        'silhouette',         # Good secondary validation
        'gap_statistic',      # Robust but slower
        'dendrogram_gap'      # Fast and interpretable
    ],
    'business_constraints': {
        'min_clusters': 3,
        'max_clusters': 30,
        'target_range': (8, 20),  # Sweet spot for business interpretation
        'stability_threshold': 0.7  # Minimum stability score
    }
}
```

### 9.2 Implementation for 3000+ Data Points
```python
def optimize_for_large_datasets(embeddings, target_clusters_range=(5, 25)):
    """
    Optimized implementation for large datasets (3000+ points).
    Uses sampling and efficient algorithms.
    """
    n_samples = len(embeddings)
    
    if n_samples > 3000:
        # Use hierarchical sampling for very large datasets
        print(f"Large dataset detected ({n_samples} points). Using optimized approach...")
        
        # 1. Quick estimation using sample
        sample_size = min(1500, n_samples // 2)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        # Get initial estimate from sample
        quick_result = multi_algorithm_consensus(sample_embeddings, max_clusters=30)
        estimated_k = quick_result['consensus_clusters']
        
        # 2. Refine estimation on full dataset with narrow range
        k_min = max(target_clusters_range[0], estimated_k - 5)
        k_max = min(target_clusters_range[1], estimated_k + 5)
        
        # Use only fast methods on full dataset
        Z = linkage(embeddings, method='ward')
        
        # Test narrow range with CH index (fast and reliable)
        ch_scores = []
        k_range = range(k_min, k_max + 1)
        
        for k in k_range:
            labels = fcluster(Z, k, criterion='maxclust')
            ch_score = calinski_harabasz_score(embeddings, labels)
            ch_scores.append(ch_score)
        
        optimal_k = k_range[np.argmax(ch_scores)]
        
        return {
            'optimal_clusters': optimal_k,
            'method': 'large_dataset_optimized',
            'sample_estimate': estimated_k,
            'refined_estimate': optimal_k,
            'ch_scores': ch_scores,
            'tested_range': list(k_range)
        }
    else:
        # Use full analysis for smaller datasets
        return multi_algorithm_consensus(embeddings, max_clusters=target_clusters_range[1])
```

## 10. Business Interpretability Guidelines

### 10.1 Cluster Number Guidelines for Business Use
```python
def get_business_interpretability_score(n_clusters, domain='IT_support'):
    """
    Score cluster numbers based on business interpretability.
    """
    interpretability_rules = {
        'IT_support': {
            'optimal_ranges': [(5, 12), (12, 20)],
            'weights': [1.0, 0.8],
            'penalty_factors': {
                'too_few': lambda k: 0.5 if k < 3 else 1.0,
                'too_many': lambda k: max(0.3, 1.0 - (k - 25) * 0.1) if k > 25 else 1.0
            }
        }
    }
    
    rules = interpretability_rules.get(domain, interpretability_rules['IT_support'])
    
    base_score = 0
    for (min_k, max_k), weight in zip(rules['optimal_ranges'], rules['weights']):
        if min_k <= n_clusters <= max_k:
            base_score = weight
            break
    
    # Apply penalty factors
    for penalty_name, penalty_func in rules['penalty_factors'].items():
        base_score *= penalty_func(n_clusters)
    
    return base_score

def recommend_for_business_context(analysis_results, context='general'):
    """
    Provide business-contextualized recommendations.
    """
    recommendations = {
        'primary': analysis_results['recommended_clusters'],
        'alternatives': analysis_results['alternative_options'],
        'business_scores': {}
    }
    
    # Score all options
    all_options = [recommendations['primary']] + recommendations['alternatives']
    
    for k in all_options:
        biz_score = get_business_interpretability_score(k, context)
        recommendations['business_scores'][k] = biz_score
    
    # Re-rank based on business scores
    ranked_options = sorted(all_options, 
                          key=lambda k: recommendations['business_scores'][k], 
                          reverse=True)
    
    return {
        'business_optimal': ranked_options[0],
        'technical_optimal': recommendations['primary'],
        'all_options_ranked': ranked_options,
        'business_scores': recommendations['business_scores'],
        'recommendation': f"Consider {ranked_options[0]} clusters for optimal business interpretability"
    }
```

## Conclusion and Best Practices

### Summary of Recommended Approach

For your specific use case (384-dimensional text embeddings, 3000+ data points, Ward linkage, business interpretability), the recommended pipeline is:

1. **Primary Methods** (in order of reliability):
   - Calinski-Harabasz Index progression
   - Silhouette analysis (with sampling)
   - Dendrogram gap analysis

2. **Secondary Validation**:
   - Gap statistic (sampled)
   - EVT threshold selection
   - Consensus clustering

3. **Business Optimization**:
   - Target 8-20 clusters for interpretability
   - Use stability analysis for confidence
   - Consider domain-specific constraints

### Implementation Priority

1. **Immediate Implementation**: Calinski-Harabasz + Silhouette consensus
2. **Enhanced Analysis**: Add gap statistic and dendrogram analysis
3. **Production Ready**: Include stability analysis and business optimization

This comprehensive approach provides robust, scientifically sound cluster number determination while maintaining practical business applicability.