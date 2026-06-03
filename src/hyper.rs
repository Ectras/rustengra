use std::time::Duration;

use pyo3::{prelude::*, types::PyDict};
use rustc_hash::FxHashMap;

/// The keyword options for the cotengra Hyperoptimizer.
///
/// Unassigned options will not be passed to the function and hence the Python
/// default values will be used. Please see the cotengra documentation for details on
/// the parameters.
#[derive(Debug, Clone, Default)]
pub struct HyperOptions {
    max_time: Option<u64>,
    max_repeats: Option<usize>,
    parallel: Option<bool>,
    slicing_reconf_opts: Option<SlicingReconfOpts>,
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

    /// Sets the `slicing_reconf_opts` argument for the HyperOptimizer.
    pub fn with_slicing_reconf_opts(mut self, slicing_reconf_opts: SlicingReconfOpts) -> Self {
        self.slicing_reconf_opts = Some(slicing_reconf_opts);
        self
    }
}

macro_rules! set_opt {
    ($dict:expr, $key:expr, $value:expr) => {
        if let Some(val) = &$value {
            $dict.set_item($key, val)?;
        }
    };
}

impl<'a, 'py> IntoPyObject<'py> for &'a HyperOptions {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        set_opt!(dict, "max_repeats", self.max_repeats);
        set_opt!(dict, "max_time", self.max_time);
        set_opt!(dict, "parallel", self.parallel);
        set_opt!(dict, "slicing_reconf_opts", self.slicing_reconf_opts);
        Ok(dict)
    }
}

/// The dynamic slicing options passed as `slicing_reconf_opts` keyword to the
/// cotengra Hyperoptimizer.
#[derive(Debug, Clone)]
pub struct SlicingReconfOpts {
    target_size: u64,
}

impl SlicingReconfOpts {
    /// Creates new slicing reconf options.
    pub fn new(target_size: u64) -> Self {
        Self { target_size }
    }
}

impl<'a, 'py> IntoPyObject<'py> for &'a SlicingReconfOpts {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        dict.set_item("target_size", self.target_size)?;
        Ok(dict)
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
    inputs: &[Vec<usize>],
    outputs: &[usize],
    size_dict: &FxHashMap<usize, u64>,
    method: &str,
    options: &HyperOptions,
) -> PyResult<Vec<(usize, usize)>> {
    Python::initialize();
    let contraction_path = Python::attach(|py| {
        let cotengra = PyModule::import(py, "cotengra")?;

        let args = (inputs, outputs, size_dict).into_pyobject(py)?;

        let kwargs = options.into_pyobject(py)?;
        kwargs.set_item("methods", method)?;

        let opt = cotengra.call_method("HyperOptimizer", (), Some(&kwargs))?;
        opt.call_method1("search", args)?
            .call_method0("get_ssa_path")?
            .extract()
    })?;

    Ok(contraction_path)
}

#[cfg(test)]
mod tests {
    use crate::utils::validate_path;

    use super::*;

    #[test]
    fn test_hyper() {
        let inputs = [
            vec![0],
            vec![1],
            vec![0, 2],
            vec![2, 1, 3, 4],
            vec![3],
            vec![4],
        ];
        let outputs = &[];

        let size_dict = FxHashMap::from_iter([(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]);

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

        validate_path(&contraction_path);
    }

    /// Test to check if Hyperoptimization object runs in Rustengra.
    /// Due to the inherently non-deterministic nature and the short
    /// run-time, this does not return a fixed contraction path.
    /// Thus, we only check for validity of the returned path.
    #[test]
    fn test_stress_hyper() {
        let inputs = [
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7],
            vec![8],
            vec![9],
            vec![10, 0],
            vec![11, 1],
            vec![12, 2],
            vec![13, 3],
            vec![14, 4],
            vec![15, 5],
            vec![16, 6],
            vec![17, 7],
            vec![18, 8],
            vec![19, 9],
            vec![11],
            vec![18],
            vec![14],
            vec![10],
            vec![17],
            vec![13],
            vec![16],
            vec![12],
            vec![19],
            vec![15],
        ];
        let outputs = &[];

        let size_dict = FxHashMap::from_iter([
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
            (11, 2),
            (12, 2),
            (13, 2),
            (14, 2),
            (15, 2),
            (16, 2),
            (17, 2),
            (18, 2),
            (19, 2),
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
}
