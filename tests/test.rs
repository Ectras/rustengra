use std::time::Duration;

use rustc_hash::FxHashMap;
use rustengra::{
    cotengra_optimize_from_path, cotengra_optimized_greedy, cotengra_sa_tree,
    cotengra_tree_tempering,
    hyper::{cotengra_hyperoptimizer, HyperOptions},
};

#[test]
fn integration_test() {
    let inputs = [
        vec![String::from("a"), String::from("8"), String::from("9")],
        vec![String::from("5"), String::from("1"), String::from("0")],
        vec![String::from("6"), String::from("8"), String::from("9")],
        vec![String::from("4"), String::from("5"), String::from("6")],
        vec![
            String::from("0"),
            String::from("1"),
            String::from("3"),
            String::from("2"),
        ],
        vec![String::from("4"), String::from("3"), String::from("2")],
    ];
    let outputs = &[String::from("a"), String::from("6")];

    let size_dict = FxHashMap::from_iter([
        (String::from("1"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("5"), 2),
        (String::from("6"), 2),
        (String::from("7"), 2),
        (String::from("8"), 2),
        (String::from("9"), 2),
        (String::from("0"), 2),
        (String::from("a"), 2),
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
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path = cotengra_optimized_greedy(&inputs, outputs, &size_dict, 8).unwrap();
    assert_eq!(
        contraction_path,
        vec![(0, 2), (3, 6), (4, 7), (5, 8), (1, 9)]
    );
}

#[test]
fn sa_integration_test() {
    let inputs = [
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

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
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

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
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
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

    assert_eq!(
        contraction_path,
        vec![(1, 3), (4, 1), (5, 4), (0, 2), (5, 0)]
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
        vec![String::from("0")],
        vec![String::from("1")],
        vec![String::from("2")],
        vec![String::from("3")],
        vec![String::from("4")],
        vec![String::from("5")],
        vec![String::from("6")],
        vec![String::from("7")],
        vec![String::from("8")],
        vec![String::from("9")],
        vec![String::from("10"), String::from("0")],
        vec![String::from("11"), String::from("1")],
        vec![String::from("12"), String::from("2")],
        vec![String::from("13"), String::from("3")],
        vec![String::from("14"), String::from("4")],
        vec![String::from("15"), String::from("5")],
        vec![String::from("16"), String::from("6")],
        vec![String::from("17"), String::from("7")],
        vec![String::from("18"), String::from("8")],
        vec![String::from("19"), String::from("9")],
        vec![String::from("11")],
        vec![String::from("18")],
        vec![String::from("14")],
        vec![String::from("10")],
        vec![String::from("17")],
        vec![String::from("13")],
        vec![String::from("16")],
        vec![String::from("12")],
        vec![String::from("19")],
        vec![String::from("15")],
    ];
    let outputs = &[];

    let size_dict = FxHashMap::from_iter([
        (String::from("0"), 2),
        (String::from("1"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("5"), 2),
        (String::from("6"), 2),
        (String::from("7"), 2),
        (String::from("8"), 2),
        (String::from("9"), 2),
        (String::from("10"), 2),
        (String::from("11"), 2),
        (String::from("12"), 2),
        (String::from("13"), 2),
        (String::from("14"), 2),
        (String::from("15"), 2),
        (String::from("16"), 2),
        (String::from("17"), 2),
        (String::from("18"), 2),
        (String::from("19"), 2),
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
