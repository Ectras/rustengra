use std::collections::HashMap;
use std::iter::zip;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

/// Checks if Cotengra is installed in the current environment.
///
/// # Example
/// ```
/// # use rustengra::cotengra_check;
/// assert!(cotengra_check().is_ok());
/// ```
pub fn cotengra_check() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| PyModule::import(py, "cotengra").map(|_| ()))
}

/// Converts tensor leg inputs (as usize) to chars. Creates new inputs, outputs and size_dict that can be fed to Cotengra.
pub fn tensor_legs_to_digit(
    inputs: &[Vec<usize>],
    outputs: &[usize],
    size_dict: &FxHashMap<usize, u64>,
) -> (Vec<Vec<String>>, Vec<String>, FxHashMap<String, u64>) {
    let mut new_inputs = vec![Vec::new(); inputs.len()];
    let mut new_size_dict = FxHashMap::default();

    for (tensor, new_tensor) in zip(inputs.iter(), new_inputs.iter_mut()) {
        new_tensor.reserve_exact(tensor.len());
        for leg in tensor {
            let string_value = leg.to_string();
            new_tensor.push(string_value.clone());
            new_size_dict.insert(string_value, size_dict[leg]);
        }
    }
    (
        new_inputs,
        outputs.iter().map(|digit| digit.to_string()).collect(),
        new_size_dict,
    )
}

/// Accepts tensor network information and returns an optimized ContractionTree via Cotengra.
///
/// Accepts inputs as `iterable[iterable[str]]`, output as `iterable[str]`, a size dict as `dict[str, int]`,
/// a starting path as `vec![(usize, usize)]`, the subtree size for optimization as `u64` and `is_ssa` as bool.
/// Creates a `ContractionTree` in Cotengra and calls `subtree_reconfigure` to find an improved
/// Contraction. Returns a `PyResult` of the best new Contraction path converted to a `replace_path`.
/// If input !`is_ssa` converts it to an SSA path.
pub fn cotengra_optimize_from_path(
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
        opt_kwargs.set_item("inplace", true)?;

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

/// Accepts tensor network information and returns an optimized ContractionTree via Cotengra.
///
/// Accepts inputs as `iterable[iterable[char]]`, output as `iterable[char]`, a size_dict that
/// maps from `char` to u64,
/// Creates a ContractionTree in Cotengra by a Greedy method.
/// Returns a PyResult of the Greedy tree converted to a replace_path.
/// If input !`is_ssa` converts input to an SSA path.
pub fn cotengra_greedy(
    inputs: &[Vec<String>],
    outputs: Vec<String>,
    size_dict: FxHashMap<String, u64>,
) -> PyResult<Vec<(usize, usize)>> {
    pyo3::prepare_freethreaded_python();
    let contraction_path: Vec<(usize, usize)> = Python::with_gil(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let args = (inputs, outputs, size_dict).into_pyobject(py)?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("optimize", String::from("greedy"))?;

        cotengra
            .getattr("array_contract_tree")?
            .call(args, Some(&kwargs))?
            .call_method0("get_ssa_path")?
            .extract()
    })?;

    Ok(ssa_to_replace_path(contraction_path, inputs.len()))
}

/// Accepts tensor network information and returns an optimized ContractionTree via Cotengra.
///
/// Accepts inputs as `iterable[iterable[char]]`, output as `iterable[char]`, a size_dict that
/// maps from `char` to u64 and a subtree size for optimization.
/// Creates a ContractionTree in Cotengra by a Greedy method and optimizes it.
/// Returns a PyResult of the optimized tree converted to a replace_path.
/// If input !`is_ssa` converts input to an SSA path.
pub fn cotengra_optimized_greedy(
    inputs: &[Vec<String>],
    outputs: Vec<String>,
    size_dict: FxHashMap<String, u64>,
    subtree_size: usize,
) -> PyResult<Vec<(usize, usize)>> {
    pyo3::prepare_freethreaded_python();
    let contraction_path: Vec<(usize, usize)> = Python::with_gil(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let args = (inputs, outputs, size_dict).into_pyobject(py)?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("optimize", String::from("greedy"))?;

        let opt_kwargs = PyDict::new(py);
        opt_kwargs.set_item("subtree_size", subtree_size)?;
        opt_kwargs.set_item("inplace", true)?;

        cotengra
            .getattr("array_contract_tree")?
            .call(args, Some(&kwargs))?
            .call_method("subtree_reconfigure", (), Some(&opt_kwargs))?
            .call_method0("get_ssa_path")?
            .extract()
    })?;

    Ok(ssa_to_replace_path(contraction_path, inputs.len()))
}

/// Accepts tensor network information and returns an optimized ContractionTree via Cotengra.
///
/// Accepts inputs as `iterable[iterable[char]]`, output as `iterable[char]`, a size_dict that
/// maps from `char` to u64 and a subtree size for optimization.
/// Creates a ContractionTree in Cotengra by a Greedy method and optimizes it.
/// Returns a PyResult of the optimized tree converted to a replace_path.
/// If input !`is_ssa` converts input to an SSA path.
pub fn cotengra_sa_tree(
    inputs: &[Vec<String>],
    outputs: Vec<String>,
    steps: Option<usize>,
    iter: Option<usize>,
    size_dict: FxHashMap<String, u64>,
    seed: Option<u64>,
) -> PyResult<Vec<(usize, usize)>> {
    pyo3::prepare_freethreaded_python();
    let contraction_path: Vec<(usize, usize)> = Python::with_gil(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let args = (inputs, outputs, size_dict).into_pyobject(py)?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("optimize", String::from("greedy"))?;

        let tree_obj = cotengra
            .getattr("array_contract_tree")?
            .call(args, Some(&kwargs))?;
        let args = (tree_obj,).into_pyobject(py)?;

        if let Some(seed) = seed {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;

            if let Some(steps) = steps {
                kwargs.set_item("tsteps", steps)?;
            }

            if let Some(iter) = iter {
                kwargs.set_item("numiter", iter)?;
            }
            kwargs.set_item("inplace", true)?;

            cotengra
                .getattr("pathfinders")?
                .getattr("path_simulated_annealing")?
                .call_method("simulated_anneal_tree", args, Some(&kwargs))?
                .call_method0("get_ssa_path")?
                .extract()
        } else {
            cotengra
                .getattr("pathfinders")?
                .getattr("path_simulated_annealing")?
                .call_method1("simulated_anneal_tree", args)?
                .call_method0("get_ssa_path")?
                .extract()
        }
    })?;

    Ok(ssa_to_replace_path(contraction_path, inputs.len()))
}

/// Accepts tensor network information and returns an optimized ContractionTree via Cotengra.
///
/// Accepts inputs as `iterable[iterable[char]]`, output as `iterable[char]`, a size_dict that
/// maps from `char` to u64 and a subtree size for optimization.
/// Creates a ContractionTree in Cotengra by a Greedy method and optimizes it.
/// Returns a PyResult of the optimized tree converted to a replace_path.
/// If input !`is_ssa` converts input to an SSA path.
pub fn cotengra_tree_tempering(
    inputs: &[Vec<String>],
    outputs: Vec<String>,
    iter: Option<usize>,
    size_dict: FxHashMap<String, u64>,
    seed: Option<u64>,
) -> PyResult<Vec<(usize, usize)>> {
    pyo3::prepare_freethreaded_python();
    let contraction_path: Vec<(usize, usize)> = Python::with_gil(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let args = (inputs, outputs, size_dict).into_pyobject(py)?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("optimize", String::from("greedy"))?;

        let tree_obj = cotengra
            .getattr("array_contract_tree")?
            .call(args, Some(&kwargs))?;
        let args = (tree_obj,).into_pyobject(py)?;

        if let Some(seed) = seed {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;

            if let Some(iter) = iter {
                kwargs.set_item("numiter", iter)?;
            }
            kwargs.set_item("inplace", true)?;

            cotengra
                .getattr("pathfinders")?
                .getattr("path_simulated_annealing")?
                .call_method("parallel_temper_tree", args, Some(&kwargs))?
                .call_method0("get_ssa_path")?
                .extract()
        } else {
            cotengra
                .getattr("pathfinders")?
                .getattr("path_simulated_annealing")?
                .call_method1("parallel_temper_tree", args)?
                .call_method0("get_ssa_path")?
                .extract()
        }
    })?;

    Ok(ssa_to_replace_path(contraction_path, inputs.len()))
}

pub fn cotengra_hyperoptimizer(
    inputs: &[Vec<String>],
    outputs: Vec<String>,
    size_dict: FxHashMap<String, u64>,
    method: String,
    max_time: i32,
    seed: Option<u64>,
) -> PyResult<Vec<(usize, usize)>> {
    pyo3::prepare_freethreaded_python();
    let contraction_path: Vec<(usize, usize)> = Python::with_gil(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let args = (inputs, outputs, size_dict).into_pyobject(py)?;

        let kwargs = PyDict::new(py);
        if let Some(seed) = seed {
            kwargs.set_item("seed", seed)?;
        }
        kwargs.set_item("methods", method)?;
        kwargs.set_item("max_time", max_time)?;
        kwargs.set_item("parallel", true)?;

        let opt = cotengra.call_method("HyperOptimizer", (), Some(&kwargs))?;

        opt.call_method1("search", args)?
            .call_method0("get_ssa_path")?
            .extract()
    })?;

    Ok(ssa_to_replace_path(contraction_path, inputs.len()))
}

/// Converts path from SSA to replace path format.
///
/// # Example
/// ```
/// # use rustengra::ssa_to_replace_path;
/// let ssa_path = vec![(0, 3), (4, 1), (2, 5)];
/// let replace_path = ssa_to_replace_path(ssa_path, 4);
/// assert_eq!(replace_path, vec![(0, 3), (0, 1), (2, 0)]);
/// ```
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

/// Converts path from replace path to SSA path format.
///
/// # Example
/// ```
/// # use rustengra::replace_to_ssa_path;
///
/// let ssa_path = vec![(0, 3), (0, 1), (2, 0)];
/// let replace_path = replace_to_ssa_path(ssa_path, 4);
/// assert_eq!(replace_path, vec![(0, 3), (4, 1), (2, 5)]);
/// ```
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
        next_id += 1;
        *i = left_id;
        *j = right_id;
    }
    replace_path
}

#[cfg(test)]
mod tests {
    use super::*;

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
            tensor_legs_to_digit(&inputs, &outputs, &size_dict);

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
}
