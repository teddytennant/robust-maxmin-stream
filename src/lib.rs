//! Streaming algorithms for robust max-min diversification.
//!
//! Implements Gonzalez GMM, the Theorem 1 sequential solver, CoresetMMD,
//! the McCutchen and Khuller delta-Scaling subroutine, and the streaming
//! coreset (Algorithms 3-6 of Pietracaprina, Pucci, and Zanon, arXiv:2610.01456).

use std::cmp::Ordering;
use std::collections::HashSet;

/// A point in Euclidean space. Distance is L2.
pub type Point = Vec<f64>;

/// Weighted coreset returned by [`streaming_coreset`].
#[derive(Clone, Debug)]
pub struct Coreset {
    pub points: Vec<Point>,
    pub weights: Vec<f64>,
    /// Input index of each coreset point (tie breaks).
    pub indices: Vec<usize>,
    /// Threshold radius R from Algorithm 3.
    pub radius: f64,
}

/// Euclidean distance. Unequal dimensions are padded with zeros.
pub fn distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len());
    let mut acc = 0.0;
    for i in 0..n {
        let d = a.get(i).copied().unwrap_or(0.0) - b.get(i).copied().unwrap_or(0.0);
        acc += d * d;
    }
    acc.sqrt()
}

/// Minimum pairwise distance. Empty and singleton sets return +infinity.
pub fn diversity(points: &[Point]) -> f64 {
    if points.len() < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            best = best.min(distance(&points[i], &points[j]));
        }
    }
    best
}

fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

// ---------------------------------------------------------------------------
// GMM (Algorithm 1)
// ---------------------------------------------------------------------------

/// Indices chosen by Gonzalez farthest-point clustering.
///
/// The first center is the first point in input order (the paper allows an
/// arbitrary start; fixing it keeps the result deterministic). Ties on the
/// farthest point are broken by the smaller input index. If `n < k`, every
/// point is returned.
fn gmm_indices(points: &[Point], k: usize) -> Vec<usize> {
    if points.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(points.len());
    let mut chosen = Vec::with_capacity(k);
    let mut in_s = vec![false; points.len()];
    // min distance to the set selected so far
    let mut near = vec![f64::INFINITY; points.len()];

    chosen.push(0);
    in_s[0] = true;
    for i in 0..points.len() {
        if i == 0 {
            near[i] = 0.0;
        } else {
            near[i] = distance(&points[i], &points[0]);
        }
    }

    while chosen.len() < k {
        let mut best_i = None;
        let mut best_d = f64::NEG_INFINITY;
        for i in 0..points.len() {
            if in_s[i] {
                continue;
            }
            let d = near[i];
            let take = match best_i {
                None => true,
                Some(j) => d > best_d || (d == best_d && i < j),
            };
            if take {
                best_d = d;
                best_i = Some(i);
            }
        }
        let i = match best_i {
            Some(i) => i,
            None => break,
        };
        chosen.push(i);
        in_s[i] = true;
        for j in 0..points.len() {
            if in_s[j] {
                continue;
            }
            near[j] = near[j].min(distance(&points[j], &points[i]));
        }
    }
    chosen
}

/// Gonzalez farthest-point (GMM). A 2-approximation for both k-center radius
/// and max-min diversification. See [`gmm_indices`] for tie breaks.
pub fn gmm(points: &[Point], k: usize) -> Vec<Point> {
    gmm_indices(points, k)
        .into_iter()
        .map(|i| points[i].clone())
        .collect()
}

fn nn_distance_in(points: &[Point], indices: &[usize], i_pos: usize) -> f64 {
    let i = indices[i_pos];
    let mut best = f64::INFINITY;
    for (t, &j) in indices.iter().enumerate() {
        if t == i_pos {
            continue;
        }
        best = best.min(distance(&points[i], &points[j]));
    }
    best
}

fn nn_distance_full(points: &[Point], i: usize) -> f64 {
    let mut best = f64::INFINITY;
    for j in 0..points.len() {
        if j == i {
            continue;
        }
        best = best.min(distance(&points[i], &points[j]));
    }
    best
}

/// The `z` indices of `pool` with largest nearest-neighbor distance.
///
/// `nn` is evaluated on the original point set. Ties: smaller input index
/// ranks as more outlier (removed first).
fn top_outliers(points: &[Point], pool: &[usize], z: usize, nn_inside_pool: bool) -> HashSet<usize> {
    let mut ranked: Vec<(f64, usize)> = pool
        .iter()
        .enumerate()
        .map(|(pos, &idx)| {
            let d = if nn_inside_pool {
                nn_distance_in(points, pool, pos)
            } else {
                nn_distance_full(points, idx)
            };
            (d, idx)
        })
        .collect();
    ranked.sort_by(|a, b| cmp_f64(b.0, a.0).then(a.1.cmp(&b.1)));
    let n_remove = z.min(ranked.len());
    ranked.into_iter().take(n_remove).map(|(_, idx)| idx).collect()
}

// ---------------------------------------------------------------------------
// Theorem 1 sequential solver
// ---------------------------------------------------------------------------

/// 2-approximation for robust max-min diversification under the paper's
/// separation assumption.
///
/// `S = GMM(X, k+z)`. Outliers are the `z` points of `S` with largest
/// nearest-neighbor distance inside `S` (ties: smaller input index). Returns
/// `S` without those outliers.
pub fn sequential_mmd(points: &[Point], k: usize, z: usize) -> Vec<Point> {
    if points.is_empty() || k == 0 {
        return Vec::new();
    }
    let s = gmm_indices(points, k.saturating_add(z));
    let outliers = top_outliers(points, &s, z, true);
    s.into_iter()
        .filter(|i| !outliers.contains(i))
        .map(|i| points[i].clone())
        .collect()
}

// ---------------------------------------------------------------------------
// CoresetMMD (Algorithm 2)
// ---------------------------------------------------------------------------

fn max_dist_to(points: &[Point], centers: &[usize]) -> f64 {
    if centers.is_empty() {
        return if points.is_empty() { 0.0 } else { f64::INFINITY };
    }
    let mut worst = 0.0;
    for (i, p) in points.iter().enumerate() {
        if centers.iter().any(|&c| c == i) {
            continue;
        }
        let d = centers
            .iter()
            .map(|&c| distance(p, &points[c]))
            .fold(f64::INFINITY, f64::min);
        if d > worst {
            worst = d;
        }
    }
    worst
}

fn farthest_from(points: &[Point], selected: &[bool]) -> Option<(usize, f64)> {
    let mut best_i = None;
    let mut best_d = f64::NEG_INFINITY;
    for i in 0..points.len() {
        if selected[i] {
            continue;
        }
        let d = if !selected.iter().any(|&s| s) {
            f64::INFINITY
        } else {
            let mut near = f64::INFINITY;
            for (j, &on) in selected.iter().enumerate() {
                if on {
                    near = near.min(distance(&points[i], &points[j]));
                }
            }
            near
        };
        let take = match best_i {
            None => true,
            Some(j) => d > best_d || (d == best_d && i < j),
        };
        if take {
            best_d = d;
            best_i = Some(i);
        }
    }
    best_i.map(|i| (i, best_d))
}

/// CoresetMMD with a caller-supplied diversification solver `A`.
///
/// `A` receives the inlier coreset and `k`, and should return up to `k` points.
pub fn coreset_mmd_with<A>(points: &[Point], k: usize, z: usize, eps: f64, solver: A) -> Vec<Point>
where
    A: Fn(&[Point], usize) -> Vec<Point>,
{
    if points.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut selected = vec![false; points.len()];
    let mut t_idx = gmm_indices(points, k.saturating_add(z));
    for &i in &t_idx {
        selected[i] = true;
    }
    // Outliers are frozen from the initial GMM set. NN is in the full X.
    let outliers = top_outliers(points, &t_idx, z, false);
    let r = max_dist_to(points, &t_idx);
    let eps_prime = eps / (1.0 + eps);
    let thresh = (eps_prime / 2.0) * r;
    while t_idx.len() < points.len() {
        let (far_i, far_d) = match farthest_from(points, &selected) {
            Some(v) => v,
            None => break,
        };
        if far_d <= thresh {
            break;
        }
        selected[far_i] = true;
        t_idx.push(far_i);
    }
    let kept: Vec<Point> = t_idx
        .into_iter()
        .filter(|i| !outliers.contains(i))
        .map(|i| points[i].clone())
        .collect();
    solver(&kept, k)
}

/// CoresetMMD using GMM as the default 2-approximate solver `A`.
///
/// Under separation this is a `2(1+eps)`-approximation.
pub fn coreset_mmd(points: &[Point], k: usize, z: usize, eps: f64) -> Vec<Point> {
    coreset_mmd_with(points, k, z, eps, gmm)
}

// ---------------------------------------------------------------------------
// delta-Scaling (McCutchen and Khuller 2008, Section 2)
// ---------------------------------------------------------------------------

/// Choose `(alpha, m)` so the McCutchen factor
/// `2 * (1 + 1/(alpha-1)) * alpha^(1/m)` is at most `2+delta`.
///
/// For `delta >= 0.5`, `alpha = 8` and `m = 10` is enough (factor about 2.81).
/// `m` is clamped to at least 1 and `alpha` to at least 2.
fn scaling_parameters(delta: f64) -> (f64, usize) {
    if delta >= 0.5 {
        return (8.0, 10);
    }
    let target = 2.0 + delta.max(0.0);
    // 2(1 + 1/(alpha-1)) = 2 + delta/2 when alpha-1 = 4/delta, then pick m
    // so the remaining alpha^(1/m) fits under 2+delta.
    let mut alpha = if delta > 1e-9 {
        (1.0 + 4.0 / delta).max(2.0)
    } else {
        2.0
    };
    let mut m: usize = 1;
    for _ in 0..200_000 {
        let factor = mc_factor(alpha, m);
        if factor <= target + 1e-9 {
            return (alpha.max(2.0), m.max(1));
        }
        if m >= 20_000 {
            alpha = (alpha * 1.5).max(alpha + 1.0);
            m = 1;
            if alpha > 1.0e6 {
                return (alpha, 20_000);
            }
        } else {
            m += 1;
        }
    }
    (alpha.max(2.0), m.max(1))
}

fn mc_factor(alpha: f64, m: usize) -> f64 {
    let m = m.max(1) as f64;
    2.0 * (1.0 + 1.0 / (alpha - 1.0)) * alpha.powf(1.0 / m)
}

struct ScaleInstance {
    /// Lower bound on the optimal k-center radius. Raised by exactly `alpha`.
    lower: f64,
    /// Centers, in the order they were kept (input order among survivors).
    centers: Vec<(usize, Point)>,
}

impl ScaleInstance {
    fn covering_radius(&self, eta: f64) -> f64 {
        eta * self.lower
    }

    fn dist_to_centers(&self, p: &Point) -> f64 {
        self.centers
            .iter()
            .map(|(_, c)| distance(p, c))
            .fold(f64::INFINITY, f64::min)
    }

    fn covered(&self, p: &Point, eta: f64) -> bool {
        if self.centers.is_empty() {
            return false;
        }
        self.dist_to_centers(p) <= self.covering_radius(eta)
    }

    /// Drop centers closer than `2 * alpha * lower` to an earlier survivor.
    fn merge(&mut self, alpha: f64) {
        let min_sep = 2.0 * alpha * self.lower;
        let old = std::mem::take(&mut self.centers);
        for (idx, c) in old {
            let too_close = self.centers.iter().any(|(_, s)| distance(&c, s) < min_sep);
            if !too_close {
                self.centers.push((idx, c));
            }
        }
    }

    fn insert(&mut self, idx: usize, p: &Point, k: usize, alpha: f64, eta: f64) {
        // A finite number of raises suffices: each multiplies the lower bound
        // by alpha, and eta * lower eventually covers any finite distance.
        for _ in 0..10_000 {
            if self.covered(p, eta) {
                return;
            }
            if self.centers.len() < k {
                self.centers.push((idx, p.clone()));
                return;
            }
            // Covering is invalid: k centers, separated by at least 2*alpha*lower,
            // and p farther than eta*lower from all of them. That proves OPT >= alpha*lower.
            self.lower *= alpha;
            if !self.lower.is_finite() {
                return;
            }
            self.merge(alpha);
        }
    }
}

struct ScalingState {
    k: usize,
    alpha: f64,
    eta: f64,
    m: usize,
    /// Distinct points gathered until k+1 of them define r0. Length at most k.
    pending: Vec<(usize, Point)>,
    r0: Option<f64>,
    instances: Vec<ScaleInstance>,
    solution_centers: Vec<Point>,
    solution_radius: f64,
}

impl ScalingState {
    fn new(k: usize, delta: f64) -> Self {
        let (alpha, m) = scaling_parameters(delta);
        let eta = 2.0 * alpha * alpha / (alpha - 1.0);
        Self {
            k,
            alpha,
            eta,
            m,
            pending: Vec::new(),
            r0: None,
            instances: Vec::new(),
            solution_centers: Vec::new(),
            solution_radius: 0.0,
        }
    }

    fn min_pairwise(pts: &[(usize, Point)]) -> f64 {
        let mut best = f64::INFINITY;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                best = best.min(distance(&pts[i].1, &pts[j].1));
            }
        }
        best
    }

    fn is_duplicate(&self, p: &Point) -> bool {
        self.pending.iter().any(|(_, q)| distance(q, p) == 0.0)
    }

    fn refresh_pending_solution(&mut self) {
        self.solution_centers = self.pending.iter().map(|(_, p)| p.clone()).collect();
        self.solution_radius = 0.0;
    }

    fn spawn(&mut self, seed: &[(usize, Point)]) {
        let r0 = match self.r0 {
            Some(v) => v,
            None => return,
        };
        self.instances = (0..self.m)
            .map(|i| {
                // Instance i (i = 0..m-1) starts at r = r0 * alpha^((i/m) - 1).
                let exp = (i as f64) / (self.m as f64) - 1.0;
                ScaleInstance {
                    lower: r0 * self.alpha.powf(exp),
                    centers: Vec::new(),
                }
            })
            .collect();
        for (idx, p) in seed {
            self.feed_instances(*idx, p);
        }
        self.refresh_instance_solution();
    }

    fn feed_instances(&mut self, idx: usize, p: &Point) {
        let k = self.k;
        let alpha = self.alpha;
        let eta = self.eta;
        for inst in &mut self.instances {
            inst.insert(idx, p, k, alpha, eta);
        }
    }

    fn refresh_instance_solution(&mut self) {
        let eta = self.eta;
        // Smallest solution radius (eta * lower). Ties: lowest instance index.
        let best = self.instances.iter().enumerate().min_by(|(_, a), (_, b)| {
            cmp_f64(eta * a.lower, eta * b.lower)
        });
        if let Some((_, inst)) = best {
            self.solution_centers = inst.centers.iter().map(|(_, p)| p.clone()).collect();
            self.solution_radius = eta * inst.lower;
        }
    }

    fn push(&mut self, idx: usize, p: &Point) {
        if self.k == 0 {
            return;
        }
        if self.r0.is_none() {
            if self.is_duplicate(p) {
                return;
            }
            self.pending.push((idx, p.clone()));
            if self.pending.len() == self.k + 1 {
                let seed = self.pending.clone();
                let gap = Self::min_pairwise(&seed);
                // r0 is half the minimum pairwise distance among the first k+1
                // distinct points. Duplicate pairs are already excluded, so the
                // gap is positive. If it is not, every point coincides.
                if !gap.is_finite() || gap == 0.0 {
                    self.pending.truncate(1);
                    self.refresh_pending_solution();
                    return;
                }
                self.r0 = Some(0.5 * gap);
                self.pending.clear();
                self.spawn(&seed);
            } else {
                self.refresh_pending_solution();
            }
            return;
        }
        self.feed_instances(idx, p);
        self.refresh_instance_solution();
    }

    fn solution(&self) -> (Vec<Point>, f64) {
        (self.solution_centers.clone(), self.solution_radius)
    }
}

/// `(2+delta)`-approximate streaming k-center (McCutchen and Khuller).
///
/// Returns centers and a covering radius `r` with `d(x, S) <= r` for every
/// input point and `|S| <= k`. Initialization peeks at the first `k+1` distinct
/// points to set `r0`, then processes the whole stream, including those points.
/// Duplicate points (distance 0) are ignored when computing `r0`. If every
/// point coincides, the result is a single center and radius 0.
pub fn scaling_k_center(points: &[Point], k: usize, delta: f64) -> (Vec<Point>, f64) {
    if points.is_empty() || k == 0 {
        return (Vec::new(), 0.0);
    }
    if k >= points.len() {
        return (points.to_vec(), 0.0);
    }
    let mut state = ScalingState::new(k, delta);
    for (i, p) in points.iter().enumerate() {
        state.push(i, p);
    }
    let (centers, rad) = state.solution();
    if centers.is_empty() {
        return (vec![points[0].clone()], 0.0);
    }
    let mut max_d = 0.0;
    for p in points {
        let d = centers
            .iter()
            .map(|c| distance(p, c))
            .fold(f64::INFINITY, f64::min);
        if d > max_d {
            max_d = d;
        }
    }
    // eta * lower is the analysis radius. Take the true covering distance as
    // well so a floating-point shortfall cannot report an infeasible radius.
    (centers, rad.max(max_d))
}

// ---------------------------------------------------------------------------
// StreamingCoreset (Algorithms 3, 4, 5)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Slot {
    index: usize,
    point: Point,
    weight: f64,
}

struct CoresetBuilder {
    slots: Vec<Slot>,
}

impl CoresetBuilder {
    fn new() -> Self {
        Self { slots: Vec::new() }
    }

    fn nearest(&self, p: &Point) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (i, s) in self.slots.iter().enumerate() {
            let d = distance(p, &s.point);
            if d < best_d || (d == best_d && s.index < self.slots[best].index) {
                best_d = d;
                best = i;
            }
        }
        Some(best)
    }

    fn insert(&mut self, index: usize, point: &Point, threshold: f64) {
        if self.slots.is_empty() {
            self.slots.push(Slot {
                index,
                point: point.clone(),
                weight: 0.0,
            });
            return;
        }
        let p = self.nearest(point).unwrap();
        let d = distance(point, &self.slots[p].point);
        if d > threshold {
            self.slots.push(Slot {
                index,
                point: point.clone(),
                weight: 0.0,
            });
        } else {
            self.slots[p].weight += 1.0;
        }
    }

    fn shrink(&mut self, threshold: f64) {
        if self.slots.is_empty() {
            return;
        }
        let old = std::mem::take(&mut self.slots);
        let u = old
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.index)
            .map(|(i, _)| i)
            .unwrap();
        let mut kept = vec![old[u].clone()];
        let mut order: Vec<usize> = (0..old.len()).filter(|&i| i != u).collect();
        order.sort_by_key(|&i| old[i].index);
        for i in order {
            let y = &old[i];
            let mut yprime = 0;
            let mut best_d = f64::INFINITY;
            for (t, s) in kept.iter().enumerate() {
                let d = distance(&y.point, &s.point);
                if d < best_d || (d == best_d && s.index < kept[yprime].index) {
                    best_d = d;
                    yprime = t;
                }
            }
            if best_d > threshold {
                kept.push(y.clone());
            } else {
                // y' becomes the proxy for y and for every point that had y as proxy.
                kept[yprime].weight += y.weight + 1.0;
            }
        }
        self.slots = kept;
    }

    fn finish(self, radius: f64) -> Coreset {
        let mut points = Vec::with_capacity(self.slots.len());
        let mut weights = Vec::with_capacity(self.slots.len());
        let mut indices = Vec::with_capacity(self.slots.len());
        for s in self.slots {
            points.push(s.point);
            weights.push(s.weight);
            indices.push(s.index);
        }
        Coreset {
            points,
            weights,
            indices,
            radius,
        }
    }
}

/// Algorithms 3-5. `eps_prime` is the paper's ε' (not ε).
///
/// The `m` scaling instances live in the builder and each point is fed to
/// them once. `gamma = 2 * (2 + delta)`.
pub fn streaming_coreset(
    points: &[Point],
    k: usize,
    z: usize,
    eps_prime: f64,
    delta: f64,
) -> Coreset {
    if points.is_empty() {
        return Coreset {
            points: Vec::new(),
            weights: Vec::new(),
            indices: Vec::new(),
            radius: 0.0,
        };
    }
    let gamma = 2.0 * (2.0 + delta);
    let k_centers = k.saturating_add(z).max(1);
    let mut scaling = ScalingState::new(k_centers, delta);
    let init_n = (k_centers + 1).min(points.len());

    let mut builder = CoresetBuilder::new();
    let thresh = |r: f64| (eps_prime / (2.0 * gamma)) * r;

    for i in 0..init_n {
        scaling.push(i, &points[i]);
    }
    let (_, mut r) = scaling.solution();
    let mut big_r = r;
    for i in 0..init_n {
        builder.insert(i, &points[i], thresh(big_r));
    }
    for i in init_n..points.len() {
        scaling.push(i, &points[i]);
        r = scaling.solution().1;
        builder.insert(i, &points[i], thresh(big_r));
        if r > 2.0 * big_r {
            big_r = r;
            builder.shrink(thresh(big_r));
        }
    }
    builder.finish(big_r)
}

// ---------------------------------------------------------------------------
// Streaming (Algorithm 6)
// ---------------------------------------------------------------------------

fn same_point(a: &Point, b: &Point) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn nn_among(points: &[Point], i: usize) -> f64 {
    let mut best = f64::INFINITY;
    for j in 0..points.len() {
        if j == i {
            continue;
        }
        best = best.min(distance(&points[i], &points[j]));
    }
    best
}

/// Streaming robust max-min diversification (Algorithm 6) with GMM as `A`.
///
/// Returns exactly `k` points when `n >= k + z` and `k >= 1`.
///
/// The paper assumes separation, under which `T` has at least `z` zero-weight
/// points and those with the largest nearest-neighbor distance inside `T` are
/// the outliers. Fallback: if fewer than `z` zero-weight points exist, take
/// all of them and fill from the smallest-weight points (tie: smaller input
/// index). Removal stops early if it would leave fewer than `min(k, n - z)`
/// points, and the solution is padded from the stream so it still has that
/// size when enough non-outlier points exist.
pub fn streaming_mmd(points: &[Point], k: usize, z: usize, eps: f64, delta: f64) -> Vec<Point> {
    streaming_mmd_with(points, k, z, eps, delta, gmm)
}

/// Algorithm 6 with a caller-supplied solver `A`.
pub fn streaming_mmd_with<A>(
    points: &[Point],
    k: usize,
    z: usize,
    eps: f64,
    delta: f64,
    solver: A,
) -> Vec<Point>
where
    A: Fn(&[Point], usize) -> Vec<Point>,
{
    if points.is_empty() || k == 0 {
        return Vec::new();
    }
    let n = points.len();
    let need = k.min(n.saturating_sub(z));
    let eps_prime = eps / (1.0 + eps);
    let coreset = streaming_coreset(points, k, z, eps_prime, delta);
    let t = &coreset.points;

    let mut zero: Vec<usize> = (0..t.len()).filter(|&i| coreset.weights[i] == 0.0).collect();
    zero.sort_by(|&a, &b| {
        cmp_f64(nn_among(t, b), nn_among(t, a)).then(coreset.indices[a].cmp(&coreset.indices[b]))
    });

    let mut rest: Vec<usize> = (0..t.len()).filter(|&i| coreset.weights[i] != 0.0).collect();
    rest.sort_by(|&a, &b| {
        cmp_f64(coreset.weights[a], coreset.weights[b])
            .then(coreset.indices[a].cmp(&coreset.indices[b]))
    });

    let mut outlier_at = Vec::new();
    let can_remove = if t.len() >= need {
        z.min(t.len() - need)
    } else {
        z.min(t.len())
    };
    for i in zero.into_iter().chain(rest.into_iter()) {
        if outlier_at.len() >= can_remove || outlier_at.len() >= z {
            break;
        }
        outlier_at.push(i);
    }
    let outlier_set: HashSet<usize> = outlier_at.iter().copied().collect();
    let outlier_pts: Vec<Point> = outlier_at.iter().map(|&i| t[i].clone()).collect();

    let kept: Vec<Point> = t
        .iter()
        .enumerate()
        .filter(|(i, _)| !outlier_set.contains(i))
        .map(|(_, p)| p.clone())
        .collect();
    let mut sol = solver(&kept, k);

    if sol.len() < need {
        for p in points {
            if sol.len() >= need {
                break;
            }
            if sol.iter().any(|s| same_point(s, p)) {
                continue;
            }
            if outlier_pts.iter().any(|o| same_point(o, p)) {
                continue;
            }
            sol.push(p.clone());
        }
    }
    if sol.len() < need {
        for p in points {
            if sol.len() >= need {
                break;
            }
            if sol.iter().any(|s| same_point(s, p)) {
                continue;
            }
            sol.push(p.clone());
        }
    }
    if sol.len() > k {
        sol.truncate(k);
    }
    sol
}

