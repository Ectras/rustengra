use rustc_hash::FxHashMap;

/// Converts path from SSA to replace left path format.
///
/// # Example
/// ```
/// # use rustengra::utils::ssa_to_replace_path;
/// let ssa_path = vec![(0, 3), (4, 1), (2, 5)];
/// let replace_path = ssa_to_replace_path(ssa_path, 4);
/// assert_eq!(replace_path, vec![(0, 3), (0, 1), (2, 0)]);
/// ```
pub fn ssa_to_replace_path(
    mut ssa_path: Vec<(usize, usize)>,
    tensor_len: usize,
) -> Vec<(usize, usize)> {
    let mut id_update = FxHashMap::default();
    for (next_id, (i, j)) in (tensor_len..).zip(&mut ssa_path) {
        let left_id = *id_update.get(i).unwrap_or(i);
        let right_id = *id_update.get(j).unwrap_or(j);

        id_update.insert(next_id, left_id);
        *i = left_id;
        *j = right_id;
    }
    ssa_path
}

/// Converts path from replace left path format to SSA path format.
///
/// # Example
/// ```
/// # use rustengra::utils::replace_to_ssa_path;
///
/// let ssa_path = vec![(0, 3), (0, 1), (2, 0)];
/// let replace_path = replace_to_ssa_path(ssa_path, 4);
/// assert_eq!(replace_path, vec![(0, 3), (4, 1), (2, 5)]);
/// ```
pub fn replace_to_ssa_path(
    mut replace_path: Vec<(usize, usize)>,
    tensor_len: usize,
) -> Vec<(usize, usize)> {
    let mut id_update = FxHashMap::default();
    for (next_id, (i, j)) in (tensor_len..).zip(&mut replace_path) {
        let left_id = *id_update.get(i).unwrap_or(i);
        let right_id = *id_update.get(j).unwrap_or(j);

        id_update.insert(*i, next_id);
        *i = left_id;
        *j = right_id;
    }
    replace_path
}
