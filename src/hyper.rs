use std::time::Duration;

use pyo3::{prelude::*, types::PyDict};
use rustc_hash::FxHashMap;

use crate::utils::ssa_to_replace_path;

/// The keyword options for the cotengra Hyperoptimizer.
///
/// Unassigned options will not be passed to the function and hence the Python
/// default values will be used. Please see the cotengra documentation for details on
/// the parameters.
#[derive(Default)]
pub struct HyperOptions {
    max_time: Option<u64>,
    max_repeats: Option<usize>,
    parallel: Option<bool>,
}

impl HyperOptions {
    /// Creates the default HyperOptimizer options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `max_time` argument for the HyperOptimizer.
    pub fn with_max_time(mut self, time: &Duration) -> Self {
        self.max_time = Some(time.as_secs());
        self
    }

    /// Sets the `max_repeats` argument for the HyperOptimizer.
    pub fn with_max_repeats(mut self, repeats: usize) -> Self {
        self.max_repeats = Some(repeats);
        self
    }

    /// Sets the `parallel` argument for the HyperOptimizer.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }
}

/// Runs the Hyperoptimizer of cotengra on the given inputs. Additional inputs to the
/// Hyperoptimizer can be passed with the [`HyperOptions`] struct.
///
/// # Python Dependency
/// Python 3 must be installed with `cotengra` and `kahypar` packages installed.
/// Can also work with virtual environments if the binary is run from a terminal with
/// actived virtual environment.
pub fn cotengra_hyperoptimizer(
    inputs: &[Vec<String>],
    outputs: &[String],
    size_dict: &FxHashMap<String, u64>,
    method: &str,
    options: &HyperOptions,
) -> PyResult<Vec<(usize, usize)>> {
    Python::initialize();
    let contraction_path: Vec<(usize, usize)> = Python::attach(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let args = (inputs, outputs, size_dict).into_pyobject(py)?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("methods", method)?;
        if let Some(max_repeats) = options.max_repeats {
            kwargs.set_item("max_repeats", max_repeats)?;
        }
        if let Some(max_time) = options.max_time {
            kwargs.set_item("max_time", max_time)?;
        }
        if let Some(parallel) = options.parallel {
            kwargs.set_item("parallel", parallel)?;
        }

        let opt = cotengra.call_method("HyperOptimizer", (), Some(&kwargs))?;
        opt.call_method1("search", args)?
            .call_method0("get_ssa_path")?
            .extract()
    })?;

    Ok(ssa_to_replace_path(contraction_path, inputs.len()))
}
