/// Validates that the contraction path is valid, i.e. that it does not contract
/// already contracted tensors.
pub fn validate_path(path: &[(usize, usize)]) {
    let mut contracted = Vec::with_capacity(path.len());
    for (u, v) in path {
        assert!(
            !contracted.contains(u),
            "Contracting already contracted tensors: {u:?}, path: {path:?}"
        );
        contracted.push(*v);
    }
}
