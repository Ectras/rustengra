use std::iter::zip;

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

/// Converts tensor leg inputs (as usize) to chars. Creates new inputs, outputs and size_dict that can be fed to Cotengra.
pub fn tensor_legs_to_digit(
    inputs: &[Vec<usize>],
    outputs: &[usize],
    size_dict: &FxHashMap<usize, u64>,
) -> (Vec<Vec<char>>, Vec<char>, FxHashMap<char, u64>) {
    let mut new_inputs = vec![Vec::new(); inputs.len()];
    let mut new_size_dict = FxHashMap::default();

    for (tensor, new_tensor) in zip(inputs.iter(), new_inputs.iter_mut()) {
        new_tensor.reserve_exact(tensor.len());
        for leg in tensor {
            let string_value = get_symbol(*leg);
            new_tensor.push(string_value.clone());
            new_size_dict.insert(string_value, size_dict[leg]);
        }
    }
    (
        new_inputs,
        outputs.iter().copied().map(get_symbol).collect(),
        new_size_dict,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_inputs_to_string() {
        let inputs = vec![vec![1505, 0, 2, 1], vec![4, 3, 2, 1], vec![4, 3, 5, 6]];
        let outputs = vec![5, 6];
        let size_dict = FxHashMap::from_iter([
            (1505, 4),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
            (5, 10),
            (6, 11),
        ]);

        let (new_inputs, new_outputs, new_size_dict) =
            tensor_legs_to_digit(&inputs, &outputs, &size_dict);

        assert_eq!(
            new_inputs,
            vec![
                vec!['\u{066D}', 'a', 'c', 'b'],
                vec!['e', 'd', 'c', 'b'],
                vec!['e', 'd', 'f', 'g']
            ]
        );
        assert_eq!(new_outputs, vec!['f', 'g']);
        assert_eq!(
            new_size_dict,
            FxHashMap::from_iter([
                ('\u{066D}', 4),
                ('a', 5),
                ('b', 6),
                ('c', 7),
                ('d', 8),
                ('e', 9),
                ('f', 10),
                ('g', 11),
            ])
        );
    }
}
