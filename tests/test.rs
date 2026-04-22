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
        vec!['k', 'i', 'j'],
        vec!['f', 'b', 'a'],
        vec!['g', 'i', 'j'],
        vec!['e', 'f', 'g'],
        vec!['a', 'b', 'd', 'c'],
        vec!['e', 'd', 'c'],
    ];
    let outputs = &['k', 'g'];

    let size_dict = FxHashMap::from_iter([
        ('a', 2),
        ('b', 2),
        ('c', 2),
        ('d', 2),
        ('e', 2),
        ('f', 2),
        ('g', 2),
        ('h', 2),
        ('i', 2),
        ('j', 2),
        ('k', 2),
    ]);

    let ssa_path = vec![(0, 1), (6, 2), (7, 3), (8, 4), (9, 5)];

    let contraction_path =
        cotengra_optimize_from_path(&inputs, outputs, &size_dict, ssa_path, 8, true).unwrap();
    assert_eq!(
        contraction_path,
        vec![(4, 5), (1, 6), (3, 7), (0, 2), (8, 9)]
    );
}

#[test]
fn optimized_greedy_integration_test() {
    let inputs = [
        vec!['a'],
        vec!['b'],
        vec!['a', 'c'],
        vec!['c', 'b', 'd', 'e'],
        vec!['d'],
        vec!['e'],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)]);

    let contraction_path = cotengra_optimized_greedy(&inputs, outputs, &size_dict, 8).unwrap();
    assert_eq!(
        contraction_path,
        vec![(0, 2), (3, 6), (4, 7), (5, 8), (1, 9)]
    );
}

#[test]
fn sa_integration_test() {
    let inputs = [
        vec!['a'],
        vec!['b'],
        vec!['a', 'c'],
        vec!['c', 'b', 'd', 'e'],
        vec!['d'],
        vec!['e'],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)]);

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
        vec!['a'],
        vec!['b'],
        vec!['a', 'c'],
        vec!['c', 'b', 'd', 'e'],
        vec!['d'],
        vec!['e'],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)]);

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
        vec!['a'],
        vec!['b'],
        vec!['a', 'c'],
        vec!['c', 'b', 'd', 'e'],
        vec!['d'],
        vec!['e'],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 2)]);

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
        vec!['a'],
        vec!['b'],
        vec!['c'],
        vec!['d'],
        vec!['e'],
        vec!['f'],
        vec!['g'],
        vec!['h'],
        vec!['i'],
        vec!['j'],
        vec!['k', 'a'],
        vec!['l', 'b'],
        vec!['m', 'c'],
        vec!['n', 'd'],
        vec!['o', 'e'],
        vec!['p', 'f'],
        vec!['q', 'g'],
        vec!['r', 'h'],
        vec!['s', 'i'],
        vec!['t', 'j'],
        vec!['l'],
        vec!['s'],
        vec!['o'],
        vec!['k'],
        vec!['r'],
        vec!['n'],
        vec!['q'],
        vec!['m'],
        vec!['t'],
        vec!['p'],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([
        ('a', 2),
        ('b', 2),
        ('c', 2),
        ('d', 2),
        ('e', 2),
        ('f', 2),
        ('g', 2),
        ('h', 2),
        ('i', 2),
        ('j', 2),
        ('k', 2),
        ('l', 2),
        ('m', 2),
        ('n', 2),
        ('o', 2),
        ('p', 2),
        ('q', 2),
        ('r', 2),
        ('s', 2),
        ('t', 2),
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
