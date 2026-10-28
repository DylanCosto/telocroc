//! Hierarchical clustering as SciPy implements it: `linkage` (nearest-neighbour chain for
//! ward, minimum spanning tree for single) and `fcluster(criterion='distance')`.

/// Linkage criterion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Single,
    Ward,
}

/// One merge step: the two cluster ids (leaves are `0..n`, merges `n..`) and the height.
#[derive(Clone, Copy, Debug)]
struct Link {
    a: usize,
    b: usize,
    dist: f64,
}

fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    n * i - (i * (i + 1)) / 2 + (j - i - 1)
}

/// `scipy.cluster.hierarchy.linkage(condensed, method)`, returned in SciPy's sorted,
/// relabelled order (columns 0-2 of Z).
fn linkage(condensed: &[f64], n: usize, method: Method) -> Vec<Link> {
    let mut z = match method {
        Method::Single => mst_single(condensed, n),
        Method::Ward => nn_chain(condensed, n),
    };
    // np.argsort(Z[:, 2], kind='mergesort') is stable
    z.sort_by(|x, y| x.dist.total_cmp(&y.dist));
    relabel(&mut z, n);
    z
}

fn nn_chain(condensed: &[f64], n: usize) -> Vec<Link> {
    let mut d = condensed.to_vec();
    let mut size = vec![1usize; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n - 1);
    for _ in 0..n - 1 {
        if chain.is_empty() {
            chain.push(size.iter().position(|&s| s > 0).expect("an active cluster remains"));
        }
        let (mut x, mut y, mut current_min);
        loop {
            x = *chain.last().expect("chain is not empty");
            if chain.len() > 1 {
                y = chain[chain.len() - 2];
                current_min = d[condensed_index(n, x, y)];
            } else {
                y = usize::MAX;
                current_min = f64::INFINITY;
            }
            for i in 0..n {
                if size[i] == 0 || i == x {
                    continue;
                }
                let dist = d[condensed_index(n, x, i)];
                if dist < current_min {
                    current_min = dist;
                    y = i;
                }
            }
            if chain.len() > 1 && y == chain[chain.len() - 2] {
                break;
            }
            chain.push(y);
        }
        chain.truncate(chain.len() - 2);
        if x > y {
            std::mem::swap(&mut x, &mut y);
        }
        let (nx, ny) = (size[x], size[y]);
        z.push(Link { a: x, b: y, dist: current_min });
        size[x] = 0;
        size[y] = nx + ny;
        for i in 0..n {
            let ni = size[i];
            if ni == 0 || i == y {
                continue;
            }
            let (d_xi, d_yi) = (d[condensed_index(n, i, x)], d[condensed_index(n, i, y)]);
            // SciPy's _ward update
            let t = 1.0 / (nx + ny + ni) as f64;
            d[condensed_index(n, i, y)] = ((ni + nx) as f64 * t * d_xi * d_xi + (ni + ny) as f64 * t * d_yi * d_yi
                - ni as f64 * t * current_min * current_min)
                .sqrt();
        }
    }
    z
}

fn mst_single(condensed: &[f64], n: usize) -> Vec<Link> {
    let mut d = vec![f64::INFINITY; n];
    let mut merged = vec![false; n];
    let (mut x, mut y) = (0, 0);
    let mut z = Vec::with_capacity(n - 1);
    for _ in 0..n - 1 {
        let mut current_min = f64::INFINITY;
        merged[x] = true;
        for i in 0..n {
            if merged[i] {
                continue;
            }
            let dist = condensed[condensed_index(n, x, i)];
            if d[i] > dist {
                d[i] = dist;
            }
            if d[i] < current_min {
                y = i;
                current_min = d[i];
            }
        }
        z.push(Link { a: x, b: y, dist: current_min });
        x = y;
    }
    z
}

/// SciPy's `label`: rewrite leaf/merge ids with a union-find so merge k gets id n + k.
fn relabel(z: &mut [Link], n: usize) {
    let mut parent: Vec<usize> = (0..2 * n - 1).collect();
    let find = |parent: &mut Vec<usize>, mut x: usize| {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != root {
            let next = parent[x];
            parent[x] = root;
            x = next;
        }
        root
    };
    for (k, link) in z.iter_mut().enumerate() {
        let (ra, rb) = (find(&mut parent, link.a), find(&mut parent, link.b));
        (link.a, link.b) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[ra] = n + k;
        parent[rb] = n + k;
    }
}

/// Clusters `n` items from a condensed distance vector and cuts the tree at `tree_cut`,
/// after scaling heights so the tallest merge is at most 1 (as Telogator2 does).
///
/// Returns the flat clusters as lists of item indices (each sorted), in no fixed order.
pub fn cluster(condensed: &[f64], n: usize, method: Method, tree_cut: f64) -> Vec<Vec<usize>> {
    if n <= 1 {
        return vec![(0..n).collect()];
    }
    let mut z = linkage(condensed, n, method);
    let max_distance = z.iter().map(|l| l.dist).fold(f64::NEG_INFINITY, f64::max);
    if max_distance > 1.0 {
        for l in &mut z {
            l.dist /= max_distance;
        }
    }
    // fcluster(criterion='distance'): a merge whose subtree's largest height is within the
    // cutoff becomes one flat cluster; everything below it joins that cluster.
    let mut max_dist = vec![0.0f64; n - 1];
    for (k, l) in z.iter().enumerate() {
        let mut m = l.dist;
        for child in [l.a, l.b] {
            if child >= n {
                m = m.max(max_dist[child - n]);
            }
        }
        max_dist[k] = m;
    }
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut stack = vec![(2 * n - 2, false)];
    while let Some((node, inside)) = stack.pop() {
        if node < n {
            if inside {
                if let Some(c) = clusters.last_mut() {
                    c.push(node);
                }
            } else {
                clusters.push(vec![node]);
            }
            continue;
        }
        let k = node - n;
        let starts_cluster = !inside && max_dist[k] <= tree_cut;
        if starts_cluster {
            clusters.push(Vec::new());
        }
        let inside = inside || starts_cluster;
        stack.push((z[k].b, inside));
        stack.push((z[k].a, inside));
    }
    for c in &mut clusters {
        c.sort_unstable();
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ward_cut_groups_close_points() {
        // points 0,1 close; 2,3 close; groups far apart
        let d = [0.1, 0.9, 0.95, 0.92, 0.91, 0.12];
        let mut c = cluster(&d, 4, Method::Ward, 0.5);
        c.sort();
        assert_eq!(c, vec![vec![0, 1], vec![2, 3]]);
    }
}
