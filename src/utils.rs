use rustc_hash::FxHashMap;

const BASE_SYMBOLS: &'static str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Get the symbol corresponding to `i` - runs through the usual 52 letters before
/// resorting to unicode characters, starting at `chr(192)`.
///
/// See also <https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.parser.get_symbol.html#opt_einsum.parser.get_symbol>
///
/// # Examples
/// ```
/// # use rustengra::utils::get_symbol;
/// assert_eq!(get_symbol(2), 'c');
/// assert_eq!(get_symbol(200), 'Ŕ');
/// assert_eq!(get_symbol(20000), '京');
/// ```
pub fn get_symbol(leg: usize) -> char {
    if leg < BASE_SYMBOLS.len() {
        BASE_SYMBOLS.chars().nth(leg).unwrap()
    } else {
        char::from_u32((leg + 140).try_into().unwrap()).unwrap()
    }
}

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
    let mut next_id = tensor_len;
    let mut id_update = FxHashMap::default();
    for (i, j) in &mut ssa_path {
        let left_id = *id_update.get(i).unwrap_or(i);
        let right_id = *id_update.get(j).unwrap_or(j);

        id_update.insert(next_id, left_id);
        next_id += 1;
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
    let mut next_id = tensor_len;
    let mut id_update = FxHashMap::default();
    for (i, j) in &mut replace_path {
        let left_id = *id_update.get(i).unwrap_or(i);
        let right_id = *id_update.get(j).unwrap_or(j);

        id_update.insert(*i, next_id);
        next_id += 1;
        *i = left_id;
        *j = right_id;
    }
    replace_path
}
