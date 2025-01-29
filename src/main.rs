use rustc_hash::FxHashMap;
use rustengra::cotengra_optimize_from_path;

fn main() {
    pyo3::prepare_freethreaded_python();
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

    let _ = cotengra_optimize_from_path(&inputs, outputs, size_dict, ssa_path, 8, true);
}
