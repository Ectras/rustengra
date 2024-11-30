use std::collections::HashMap;
use std::iter::zip;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

pub fn cotengra_check() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let _ = PyModule::import(py, "cotengra").unwrap();
    });
    Ok(())
}

/// Converts tensor leg inputs (as usize) to chars. Creates new inputs, outputs and size_dict that can be fed to Cotengra.
pub fn tensor_legs_to_digit(
    inputs: &[Vec<usize>],
    outputs: &[usize],
    size_dict: FxHashMap<usize, u64>,
) -> (Vec<Vec<String>>, Vec<String>, FxHashMap<String, u64>) {
    let mut new_inputs = vec![Vec::new(); inputs.len()];
    let mut leg_to_digit = FxHashMap::<usize, String>::default();
    let mut new_size_dict = FxHashMap::default();

    for (tensor, new_tensor) in zip(inputs.iter(), new_inputs.iter_mut()) {
        for leg in tensor.iter() {
            let to_string = leg_to_digit.get(leg);
            let string_value = if let Some(string_value) = to_string {
                string_value.clone()
            } else {
                leg.to_string()
            };
            leg_to_digit.insert(*leg, string_value.clone());
            new_tensor.push(string_value.clone());
            new_size_dict.insert(string_value, *size_dict.get(leg).unwrap());
        }
    }
    (
        new_inputs,
        outputs.iter().map(|digit| digit.to_string()).collect(),
        new_size_dict,
    )
}

/// Accepts tensor network information and returns an optimized ContractionTree via Cotengra
///
/// Accepts inputs as iterable[iterable[char]], output as iterable[char] and a size_dict that
/// maps from `char` to u64.
/// Creates a ContractionTree in Cotengra and calls subtree_reconfigure to find an improved
/// Contraction. Returns a PyResult of the best new Contraction path converted to a replace_path.
/// If input !`is_ssa` converts it to an SSA path.
pub fn create_and_optimize_tree(
    inputs: &[Vec<String>],
    outputs: Vec<String>,
    size_dict: FxHashMap<String, u64>,
    path: Vec<(usize, usize)>,
    subtree_size: usize,
    is_ssa: bool,
) -> PyResult<Vec<(usize, usize)>> {
    pyo3::prepare_freethreaded_python();
    let contraction_path: Vec<(usize, usize)> = Python::with_gil(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("size_dict", size_dict)?;

        let path = if !is_ssa {
            replace_to_ssa_path(path, inputs.len())
        } else {
            path
        };

        kwargs.set_item("ssa_path", path)?;

        let opt_kwargs = PyDict::new(py);
        opt_kwargs.set_item("subtree_size", subtree_size)?;

        let args = (inputs, outputs).into_pyobject(py)?;
        cotengra
            .getattr("ContractionTree")?
            .getattr("from_path")?
            .call(args, Some(&kwargs))?
            .call_method("subtree_reconfigure", (), Some(&opt_kwargs))?
            .call_method0("get_ssa_path")?
            .extract()
    })?;

    Ok(ssa_to_replace_path(contraction_path, inputs.len()))
}

/// Converts path from SSA to replace path format
pub fn ssa_to_replace_path(
    mut ssa_path: Vec<(usize, usize)>,
    tensor_len: usize,
) -> Vec<(usize, usize)> {
    let mut next_id = tensor_len;
    let mut id_update = HashMap::new();
    for (i, j) in ssa_path.iter_mut() {
        let left_id = *id_update.get(i).unwrap_or(i);
        let right_id = *id_update.get(j).unwrap_or(j);

        id_update.insert(next_id, left_id);
        next_id += 1;
        *i = left_id;
        *j = right_id;
    }
    ssa_path
}

/// Converts path from replace path to SSA path format
pub fn replace_to_ssa_path(
    mut replace_path: Vec<(usize, usize)>,
    tensor_len: usize,
) -> Vec<(usize, usize)> {
    let mut next_id = tensor_len;
    let mut id_update = HashMap::new();

    for (i, j) in replace_path.iter_mut() {
        let left_id = *id_update.get(i).unwrap_or(i);
        let right_id = *id_update.get(j).unwrap_or(j);
        id_update.insert(*i, next_id);
        *i = left_id;
        *j = right_id;

        next_id += 1;
    }
    replace_path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contengra_check() {
        let _ = cotengra_check();
    }

    #[test]
    fn test_tensor_inputs_to_string() {
        let inputs = vec![vec![0, 1, 3, 2], vec![5, 4, 3, 2], vec![5, 4, 6, 7]];
        let outputs = vec![6, 7];
        let size_dict = FxHashMap::from_iter([
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (4, 8),
            (5, 9),
            (6, 10),
            (7, 11),
        ]);

        let (new_inputs, new_outputs, new_size_dict) =
            tensor_legs_to_digit(&inputs, &outputs, size_dict);

        assert_eq!(
            new_inputs,
            vec![
                vec!["0", "1", "3", "2"],
                vec!["5", "4", "3", "2"],
                vec!["5", "4", "6", "7"]
            ]
        );
        assert_eq!(new_outputs, vec!["6", "7"]);
        assert_eq!(
            new_size_dict,
            FxHashMap::from_iter([
                (String::from("0"), 4),
                (String::from("1"), 5),
                (String::from("2"), 6),
                (String::from("3"), 7),
                (String::from("4"), 8),
                (String::from("5"), 9),
                (String::from("6"), 10),
                (String::from("7"), 11),
            ])
        );
    }

    #[test]
    fn test_ssa_to_replace_path() {
        let ssa_path = vec![(0, 3), (4, 1), (2, 5)];
        let replace_path = ssa_to_replace_path(ssa_path, 4);

        assert_eq!(replace_path, vec![(0, 3), (0, 1), (2, 0)]);
    }

    #[test]
    fn test_replace_to_ssa_path() {
        let ssa_path = vec![(0, 3), (0, 1), (2, 0)];
        let replace_path = replace_to_ssa_path(ssa_path, 4);

        assert_eq!(replace_path, vec![(0, 3), (4, 1), (2, 5)]);
    }
}
