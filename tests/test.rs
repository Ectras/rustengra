use rustc_hash::FxHashMap;
use rustengra::{cotengra_greedy, cotengra_optimize_from_path, cotengra_optimized_greedy};

#[test]
fn integration_test() {
    // pyo3::prepare_freethreaded_python();
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
    let outputs = vec![String::from("a"), String::from("6")];

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
        cotengra_optimize_from_path(&inputs, outputs, size_dict, ssa_path, 8, true).unwrap();
    assert_eq!(
        contraction_path,
        vec![(4, 5), (1, 4), (3, 1), (0, 2), (3, 0)]
    )
}

#[test]
fn optimized_greedy_integration_test() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("0")],
        vec![String::from("1")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("1"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = vec![];

    let size_dict = FxHashMap::from_iter([
        (String::from("1"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path = cotengra_optimized_greedy(&inputs, outputs, size_dict, 8).unwrap();
    assert_eq!(
        contraction_path,
        vec![(0, 2), (3, 0), (4, 3), (5, 4), (1, 5)]
    )
}

#[test]
fn greedy_integration_test() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("0")],
        vec![String::from("1")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("1"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = vec![];

    let size_dict = FxHashMap::from_iter([
        (String::from("1"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path = cotengra_greedy(&inputs, outputs, size_dict).unwrap();

    assert_eq!(
        contraction_path,
        vec![(1, 3), (4, 1), (5, 4), (0, 2), (5, 0)]
    )
}
