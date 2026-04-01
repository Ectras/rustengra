use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

use crate::utils::replace_to_ssa_path;

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
/// Contraction. Returns a `PyResult` of the best new contraction path in SSA format.
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

    Ok(contraction_path)
}

/// Accepts tensor network information and returns an optimized ContractionTree via
/// Cotengra.
///
/// Accepts inputs as `iterable[iterable[char]]`, output as `iterable[char]`, a
/// `size_dict` that maps from `char` to `u64` and a subtree size for optimization.
/// Creates a ContractionTree in Cotengra by a Greedy method and optimizes it with
/// subtree reconfiguration. Returns a PyResult of the optimized tree converted to a
/// SSA path.
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

    Ok(contraction_path)
}

/// Accepts tensor network information and returns an optimized ContractionTree via
/// Cotengra.
///
/// Accepts inputs as `iterable[iterable[char]]`, output as `iterable[char]`, a
/// `size_dict` that maps from `char` to `u64` and a subtree size for optimization.
/// Creates a ContractionTree in Cotengra by a Greedy method and optimizes it with
/// simualted annealing. Returns a PyResult of the optimized tree converted to a
/// SSA path.
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

    Ok(contraction_path)
}

/// Accepts tensor network information and returns an optimized ContractionTree via
/// Cotengra.
///
/// Accepts inputs as `iterable[iterable[char]]`, output as `iterable[char]`, a
/// `size_dict` that maps from `char` to `u64` and a subtree size for optimization.
/// Creates a ContractionTree in Cotengra by simulated annealing and optimizes it
/// using tree tempering. Returns a PyResult of the optimized tree converted to a SSA
/// path.
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

    Ok(contraction_path)
}
