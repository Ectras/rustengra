use std::iter::zip;

use rustc_hash::FxHashMap;

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
        outputs.iter().map(ToString::to_string).collect(),
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
