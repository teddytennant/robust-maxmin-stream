use robust_maxmin_stream::{
    coreset_mmd, distance, diversity, gmm, scaling_k_center, sequential_mmd, streaming_coreset,
    streaming_mmd, Point,
};

fn dist(a: &Point, b: &Point) -> f64 {
    distance(a, b)
}

fn is_outlier(p: &Point) -> bool {
    (p[0] - 100.0).abs() < 1e-9 && (p[1] - 100.0).abs() < 1e-9
        || (p[0] + 100.0).abs() < 1e-9 && (p[1] - 80.0).abs() < 1e-9
}

fn separation_instance() -> (Vec<Point>, Vec<Point>, Vec<Point>) {
    let mut inliers = Vec::new();
    let centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]];
    let offsets = [[0.0, 0.0], [0.03, 0.0], [0.0, 0.04]];
    for c in &centers {
        for o in &offsets {
            inliers.push(vec![c[0] + o[0], c[1] + o[1]]);
        }
    }
    let outliers = vec![vec![100.0, 100.0], vec![-100.0, 80.0]];
    let mut all = inliers.clone();
    all.extend(outliers.iter().cloned());
    (all, inliers, outliers)
}

fn next_combination(comb: &mut [usize], n: usize) -> bool {
    let k = comb.len();
    if k == 0 || n == 0 {
        return false;
    }
    let mut i = k;
    while i > 0 {
        i -= 1;
        if comb[i] + 1 <= n - k + i {
            comb[i] += 1;
            for j in (i + 1)..k {
                comb[j] = comb[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

fn best_div(points: &[Point], k: usize) -> f64 {
    if points.len() < k || k < 2 {
        return 0.0;
    }
    let mut comb: Vec<usize> = (0..k).collect();
    let mut best = 0.0;
    loop {
        let subset: Vec<Point> = comb.iter().map(|&i| points[i].clone()).collect();
        let d = diversity(&subset);
        if d > best {
            best = d;
        }
        if !next_combination(&mut comb, points.len()) {
            break;
        }
    }
    best
}

fn kcenter_radius(points: &[Point], center_idx: &[usize]) -> f64 {
    let mut worst = 0.0;
    for p in points {
        let mut near = f64::INFINITY;
        for &c in center_idx {
            near = near.min(dist(p, &points[c]));
        }
        if near > worst {
            worst = near;
        }
    }
    worst
}

fn opt_kcenter(points: &[Point], k: usize) -> f64 {
    let n = points.len();
    let mut comb: Vec<usize> = (0..k).collect();
    let mut best = f64::INFINITY;
    loop {
        let r = kcenter_radius(points, &comb);
        if r < best {
            best = r;
        }
        if !next_combination(&mut comb, n) {
            break;
        }
    }
    best
}

#[test]
fn gmm_line_is_two_approx() {
    let points: Vec<Point> = vec![
        vec![0.0],
        vec![1.0],
        vec![3.0],
        vec![6.0],
        vec![10.0],
        vec![15.0],
    ];
    let k = 3;
    let s = gmm(&points, k);
    assert_eq!(s.len(), k);
    let opt = best_div(&points, k);
    let got = diversity(&s);
    assert!(
        got + 1e-9 >= opt / 2.0,
        "div(S)={got} opt={opt} opt/2={}",
        opt / 2.0
    );
}

#[test]
fn sequential_on_separated_clusters() {
    let (all, _inliers, _outliers) = separation_instance();
    let sol = sequential_mmd(&all, 3, 2);
    assert_eq!(sol.len(), 3);
    assert!(sol.iter().all(|p| !is_outlier(p)), "selected an outlier: {sol:?}");
    let got = diversity(&sol);
    assert!(got >= 8.0, "div={got}, want >= 8");
}

#[test]
fn coreset_mmd_returns_inliers() {
    let (all, _, _) = separation_instance();
    let sol = coreset_mmd(&all, 3, 2, 0.5);
    assert_eq!(sol.len(), 3);
    assert!(sol.iter().all(|p| !is_outlier(p)), "selected an outlier: {sol:?}");
}

#[test]
fn scaling_is_two_plus_delta() {
    let points: Vec<Point> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.2],
        vec![0.2, 1.0],
        vec![8.0, 0.5],
        vec![9.0, 1.0],
        vec![8.5, 0.0],
        vec![4.0, 7.0],
        vec![4.5, 7.5],
        vec![3.5, 6.5],
        vec![20.0, 3.0],
    ];
    let k = 2;
    let delta = 1.0;
    let (centers, r) = scaling_k_center(&points, k, delta);
    assert!(centers.len() <= k, "|S|={} > {k}", centers.len());
    assert!(!centers.is_empty());
    for p in &points {
        let near = centers.iter().map(|c| dist(p, c)).fold(f64::INFINITY, f64::min);
        assert!(
            near <= r + 1e-8,
            "point {p:?} is {near} from centers, r={r}"
        );
    }
    let opt = opt_kcenter(&points, k);
    assert!(
        r <= (2.0 + delta) * opt + 1e-6,
        "r={r} opt={opt} bound={}",
        (2.0 + delta) * opt
    );
}

#[test]
fn streaming_coreset_properties() {
    let (all, _, outliers) = separation_instance();
    let eps_prime = 0.5;
    let delta = 1.0;
    let coreset = streaming_coreset(&all, 3, 2, eps_prime, delta);
    let gamma = 2.0 * (2.0 + delta);
    let thresh = (eps_prime / (2.0 * gamma)) * coreset.radius;
    let t = &coreset.points;
    for i in 0..t.len() {
        for j in (i + 1)..t.len() {
            let d = dist(&t[i], &t[j]);
            assert!(
                d > thresh - 1e-8,
                "pair {i},{j} distance {d} not above threshold {thresh} (R={})",
                coreset.radius
            );
        }
    }
    assert!(coreset.weights.iter().all(|&w| w >= 0.0));
    let sum_w: f64 = coreset.weights.iter().sum();
    assert!(
        (sum_w + t.len() as f64 - all.len() as f64).abs() < 1e-6,
        "sum w + |T| = {} + {} != {}",
        sum_w,
        t.len(),
        all.len()
    );
    for o in &outliers {
        let pos = t.iter().position(|p| p == o).expect("outlier missing from T");
        assert_eq!(coreset.weights[pos], 0.0, "outlier weight {}", coreset.weights[pos]);
    }
}

#[test]
fn streaming_mmd_excludes_outliers() {
    let (all, _, _) = separation_instance();
    let sol = streaming_mmd(&all, 3, 2, 0.5, 1.0);
    assert_eq!(sol.len(), 3);
    assert!(sol.iter().all(|p| !is_outlier(p)), "selected an outlier: {sol:?}");
}

#[test]
fn long_stream_coreset_stays_small() {
    let (sep, _, _) = separation_instance();
    let mut pts = sep;
    for i in 0..200 {
        let t = i as f64;
        let ang = t * 0.37;
        let rad = 0.01 * ((i % 17) as f64) / 16.0;
        pts.push(vec![20.0 + rad * ang.cos(), 20.0 + rad * ang.sin()]);
    }
    let eps_prime = 0.5;
    let delta = 1.0;
    let coreset = streaming_coreset(&pts, 3, 2, eps_prime, delta);
    assert!(
        coreset.points.len() < 40,
        "|T|={} grew with the stream (n={})",
        coreset.points.len(),
        pts.len()
    );
    let sum_w: f64 = coreset.weights.iter().sum();
    assert!((sum_w + coreset.points.len() as f64 - pts.len() as f64).abs() < 1e-6);
}
