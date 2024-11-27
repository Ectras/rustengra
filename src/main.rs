use rustc_hash::FxHashMap;
use rustengra::create_and_optimize_tree;

fn main() {
    pyo3::prepare_freethreaded_python();
    let inputs = [
        vec!['a', '8', '9'],
        vec!['5', '1', '0'],
        vec!['6', '8', '9'],
        vec!['4', '5', '6'],
        vec!['0', '1', '3', '2'],
        vec!['4', '3', '2'],
    ];
    let outputs = vec!['a', '6'];

    let size_dict = FxHashMap::from_iter([
        ('1', 2),
        ('2', 2),
        ('3', 2),
        ('4', 2),
        ('5', 2),
        ('6', 2),
        ('7', 2),
        ('8', 2),
        ('9', 2),
        ('0', 2),
        ('a', 2),
    ]);

    let ssa_path = vec![(0, 1), (6, 2), (7, 3), (8, 4), (9, 5)];

    let _ = create_and_optimize_tree(&inputs, outputs, size_dict, ssa_path, 8, true);
}
