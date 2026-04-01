use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

use crate::utils::{replace_to_ssa_path, ssa_to_replace_path};

pub mod hyper;
pub mod utils;

/// Checks if Cotengra is installed in the current environment.
///
/// # Example
/// ```
/// # use rustengra::cotengra_check;
/// assert!(cotengra_check().is_ok());
/// ```
pub fn cotengra_check() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| PyModule::import(py, "cotengra").map(|_| ()))
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
    outputs: &[String],
    size_dict: &FxHashMap<String, u64>,
    path: Vec<(usize, usize)>,
    subtree_size: usize,
    is_ssa: bool,
) -> PyResult<Vec<(usize, usize)>> {
    Python::initialize();
    let contraction_path: Vec<(usize, usize)> = Python::attach(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("size_dict", size_dict)?;

        let path = if is_ssa {
            path
        } else {
            replace_to_ssa_path(path, inputs.len())
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
/// maps from `char` to u64 and a subtree size for optimization.
/// Creates a ContractionTree in Cotengra by a Greedy method and optimizes it.
/// Returns a PyResult of the optimized tree converted to a replace_path.
/// If input !`is_ssa` converts input to an SSA path.
pub fn cotengra_optimized_greedy(
    inputs: &[Vec<String>],
    outputs: &[String],
    size_dict: &FxHashMap<String, u64>,
    subtree_size: usize,
) -> PyResult<Vec<(usize, usize)>> {
    Python::initialize();
    let contraction_path: Vec<(usize, usize)> = Python::attach(|py| {
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
    outputs: &[String],
    steps: Option<usize>,
    iter: Option<usize>,
    size_dict: &FxHashMap<String, u64>,
    seed: Option<u64>,
) -> PyResult<Vec<(usize, usize)>> {
    Python::initialize();
    let contraction_path: Vec<(usize, usize)> = Python::attach(|py| {
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
    outputs: &[String],
    iter: Option<usize>,
    size_dict: &FxHashMap<String, u64>,
    seed: Option<u64>,
) -> PyResult<Vec<(usize, usize)>> {
    Python::initialize();
    let contraction_path: Vec<(usize, usize)> = Python::attach(|py| {
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

#[cfg(test)]
mod tests {
    use crate::utils::tensor_legs_to_digit;

    use super::*;

    #[test]
    fn test_tensor_inputs_to_string() {
        let inputs = vec![vec![1505, 1, 3, 2], vec![5, 4, 3, 2], vec![5, 4, 6, 7]];
        let outputs = vec![6, 7];
        let size_dict = FxHashMap::from_iter([
            (1505, 4),
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
                vec!["1505", "1", "3", "2"],
                vec!["5", "4", "3", "2"],
                vec!["5", "4", "6", "7"]
            ]
        );
        assert_eq!(new_outputs, vec!["6", "7"]);
        assert_eq!(
            new_size_dict,
            FxHashMap::from_iter([
                (String::from("1505"), 4),
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
