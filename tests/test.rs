use std::time::Duration;

use rustc_hash::FxHashMap;
use rustengra::{
    cotengra_optimize_from_path, cotengra_optimized_greedy, cotengra_sa_tree,
    cotengra_tree_tempering,
    hyper::{cotengra_hyperoptimizer, HyperOptions},
};

#[test]
fn cotengra_optimize_from_path_test() {
    let inputs = [
        vec![10, 8, 9],
        vec![5, 1, 0],
        vec![6, 8, 9],
        vec![4, 5, 6],
        vec![0, 1, 3, 2],
        vec![4, 3, 2],
    ];
    let outputs = &[10, 6];

    let size_dict = FxHashMap::from_iter([
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (6, 2),
        (7, 2),
        (8, 2),
        (9, 2),
        (10, 2),
    ]);

    let ssa_path = vec![(0, 1), (6, 2), (7, 3), (8, 4), (9, 5)];

    let contraction_path =
        cotengra_optimize_from_path(&inputs, outputs, &size_dict, ssa_path, 8).unwrap();
    assert_eq!(
        contraction_path,
        vec![(4, 5), (1, 6), (3, 7), (0, 2), (8, 9)]
    );
}

#[test]
fn optimized_greedy_integration_test() {
    let inputs = [
        vec![0],
        vec![1],
        vec![0, 2],
        vec![2, 1, 3, 4],
        vec![3],
        vec![4],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]);

    let contraction_path = cotengra_optimized_greedy(&inputs, outputs, &size_dict, 8).unwrap();
    assert_eq!(
        contraction_path,
        vec![(0, 2), (3, 6), (4, 7), (5, 8), (1, 9)]
    );
}

#[test]
fn sa_integration_test() {
    let inputs = [
        vec![0],
        vec![1],
        vec![0, 2],
        vec![2, 1, 3, 4],
        vec![3],
        vec![4],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]);

    let contraction_path =
        cotengra_sa_tree(&inputs, outputs, None, None, &size_dict, Some(4)).unwrap();

    assert_eq!(
        contraction_path,
        vec![(4, 5), (3, 6), (1, 7), (2, 8), (0, 9)]
    );
}

#[test]
fn tempering_integration_test() {
    let inputs = [
        vec![0],
        vec![1],
        vec![0, 2],
        vec![2, 1, 3, 4],
        vec![3],
        vec![4],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]);

    let contraction_path =
        cotengra_tree_tempering(&inputs, outputs, None, &size_dict, Some(4)).unwrap();

    assert_eq!(
        contraction_path,
        vec![(1, 5), (3, 6), (4, 7), (2, 8), (0, 9)]
    );
}

#[test]
fn test_hyper() {
    let inputs = [
        vec![0],
        vec![1],
        vec![0, 2],
        vec![2, 1, 3, 4],
        vec![3],
        vec![4],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]);

    let contraction_path = cotengra_hyperoptimizer(
        &inputs,
        outputs,
        &size_dict,
        "kahypar",
        &HyperOptions::default()
            .with_max_repeats(10)
            .with_parallel(false),
    )
    .unwrap();

    assert_eq!(
        contraction_path,
        vec![(1, 3), (4, 6), (5, 7), (0, 2), (8, 9)]
    );
}

fn validate_path(path: &[(usize, usize)]) {
    let mut contracted = Vec::with_capacity(path.len());
    for (u, v) in path {
        assert!(
            !contracted.contains(u),
            "Contracting already contracted tensors: {u:?}, path: {path:?}"
        );
        contracted.push(*v);
    }
}

/// Test to check if Hyperoptimization object runs in Rustengra.
/// Due to the inherently non-deterministic nature and the short
/// run-time, this does not return a fixed contraction path.
/// Thus, we only check for validity of the returned path.
#[test]
fn test_stress_hyper() {
    let inputs = [
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![6],
        vec![7],
        vec![8],
        vec![9],
        vec![10, 0],
        vec![11, 1],
        vec![12, 2],
        vec![13, 3],
        vec![14, 4],
        vec![15, 5],
        vec![16, 6],
        vec![17, 7],
        vec![18, 8],
        vec![19, 9],
        vec![11],
        vec![18],
        vec![14],
        vec![10],
        vec![17],
        vec![13],
        vec![16],
        vec![12],
        vec![19],
        vec![15],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (6, 2),
        (7, 2),
        (8, 2),
        (9, 2),
        (10, 2),
        (11, 2),
        (12, 2),
        (13, 2),
        (14, 2),
        (15, 2),
        (16, 2),
        (17, 2),
        (18, 2),
        (19, 2),
    ]);

    let duration = Duration::from_secs(15);
    let contraction_path = cotengra_hyperoptimizer(
        &inputs,
        outputs,
        &size_dict,
        "kahypar",
        &HyperOptions::default().with_max_time(&duration),
    )
    .unwrap();

    validate_path(&contraction_path);
}
