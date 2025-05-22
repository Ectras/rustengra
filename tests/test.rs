use rustc_hash::FxHashMap;
use rustengra::{
    cotengra_greedy, cotengra_hyperoptimizer, cotengra_optimize_from_path, cotengra_optimized_greedy, cotengra_sa_tree, cotengra_tree_tempering
};

#[test]
fn integration_test() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("a"), String::from("8"), String::from("9")],
        vec![String::from("5"), String::from("1"), String::from("0")],
        vec![String::from("6"), String::from("8"), String::from("9")],
        vec![String::from("4"), String::from("5"), String::from("6")],
        vec![
            String::from("0"),
            String::from("1"),
            String::from("3"),
            String::from("2"),
        ],
        vec![String::from("4"), String::from("3"), String::from("2")],
    ];
    let outputs = vec![String::from("a"), String::from("6")];

    let size_dict = FxHashMap::from_iter([
        (String::from("1"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("5"), 2),
        (String::from("6"), 2),
        (String::from("7"), 2),
        (String::from("8"), 2),
        (String::from("9"), 2),
        (String::from("0"), 2),
        (String::from("a"), 2),
    ]);

    let ssa_path = vec![(0, 1), (6, 2), (7, 3), (8, 4), (9, 5)];

    let contraction_path =
        cotengra_optimize_from_path(&inputs, outputs, size_dict, ssa_path, 8, true).unwrap();
    assert_eq!(
        contraction_path,
        vec![(4, 5), (1, 4), (3, 1), (0, 2), (3, 0)]
    )
}

#[test]
fn optimized_greedy_integration_test() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = vec![];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path = cotengra_optimized_greedy(&inputs, outputs, size_dict, 8).unwrap();
    assert_eq!(
        contraction_path,
        vec![(0, 2), (3, 0), (4, 3), (5, 4), (1, 5)]
    )
}

#[test]
fn greedy_integration_test() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = vec![];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path = cotengra_greedy(&inputs, outputs, size_dict).unwrap();

    assert_eq!(
        contraction_path,
        vec![(1, 3), (4, 1), (5, 4), (0, 2), (5, 0)]
    )
}

#[test]
fn sa_integration_test() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = vec![];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path =
        cotengra_sa_tree(&inputs, outputs, None, None, size_dict, Some(4)).unwrap();

    assert_eq!(
        contraction_path,
        vec![(4, 5), (3, 4), (1, 3), (2, 1), (0, 2)]
    )
}

#[test]
fn tempering_integration_test() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = vec![];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path =
        cotengra_tree_tempering(&inputs, outputs, None, size_dict, Some(4)).unwrap();

    assert_eq!(
        contraction_path,
        vec![(1, 5), (3, 1), (4, 3), (2, 4), (0, 2)]
    )
}

#[test]
fn greedy_large_test() {
    let inputs = [
        vec![String::from("0"), String::from("1")],
        vec![String::from("2"), String::from("3")],
        vec![String::from("4"), String::from("5")],
        vec![String::from("6"), String::from("7")],
        vec![String::from("8"), String::from("9")],
        vec![
            String::from("10"),
            String::from("11"),
            String::from("2"),
            String::from("4"),
        ],
        vec![
            String::from("3"),
            String::from("5"),
            String::from("12"),
            String::from("13"),
        ],
        vec![
            String::from("14"),
            String::from("15"),
            String::from("11"),
            String::from("6"),
        ],
        vec![
            String::from("13"),
            String::from("7"),
            String::from("16"),
            String::from("17"),
        ],
        vec![String::from("18"), String::from("0")],
        vec![String::from("1"), String::from("19")],
        vec![String::from("20"), String::from("10")],
        vec![String::from("12"), String::from("21")],
        vec![String::from("22"), String::from("14")],
        vec![String::from("16"), String::from("23")],
        vec![String::from("24"), String::from("15")],
        vec![String::from("17"), String::from("25")],
        vec![String::from("26"), String::from("20")],
        vec![String::from("21"), String::from("27")],
        vec![String::from("28"), String::from("24")],
        vec![String::from("25"), String::from("29")],
        vec![
            String::from("30"),
            String::from("31"),
            String::from("18"),
            String::from("26"),
        ],
        vec![
            String::from("19"),
            String::from("27"),
            String::from("32"),
            String::from("33"),
        ],
        vec![
            String::from("34"),
            String::from("35"),
            String::from("31"),
            String::from("22"),
        ],
        vec![
            String::from("33"),
            String::from("23"),
            String::from("36"),
            String::from("37"),
        ],
        vec![
            String::from("38"),
            String::from("39"),
            String::from("35"),
            String::from("28"),
        ],
        vec![
            String::from("37"),
            String::from("29"),
            String::from("40"),
            String::from("41"),
        ],
        vec![String::from("42"), String::from("34")],
        vec![String::from("36"), String::from("43")],
        vec![String::from("44"), String::from("8")],
        vec![String::from("9"), String::from("45")],
        vec![
            String::from("46"),
            String::from("47"),
            String::from("42"),
            String::from("38"),
        ],
        vec![
            String::from("43"),
            String::from("40"),
            String::from("48"),
            String::from("49"),
        ],
        vec![
            String::from("50"),
            String::from("51"),
            String::from("47"),
            String::from("39"),
        ],
        vec![
            String::from("49"),
            String::from("41"),
            String::from("52"),
            String::from("53"),
        ],
        vec![
            String::from("54"),
            String::from("55"),
            String::from("51"),
            String::from("44"),
        ],
        vec![
            String::from("53"),
            String::from("45"),
            String::from("56"),
            String::from("57"),
        ],
        vec![String::from("58"), String::from("55")],
        vec![String::from("57"), String::from("59")],
        vec![String::from("30")],
        vec![String::from("32")],
        vec![String::from("46")],
        vec![String::from("48")],
        vec![String::from("50")],
        vec![String::from("52")],
        vec![String::from("54")],
        vec![String::from("56")],
        vec![String::from("58")],
        vec![String::from("59")],
    ];
    let size_dict = (0..65)
        .map(|i| (i.to_string(), 2))
        .collect::<FxHashMap<_, _>>();

    let outputs = vec![];

    let contraction_path = cotengra_greedy(&inputs, outputs, size_dict).unwrap();

    assert_eq!(
        contraction_path,
        vec![
            (22, 40),
            (10, 22),
            (12, 18),
            (10, 12),
            (0, 9),
            (21, 39),
            (0, 21),
            (10, 0),
            (1, 5),
            (2, 6),
            (1, 2),
            (11, 17),
            (1, 11),
            (10, 1),
            (36, 46),
            (38, 48),
            (36, 38),
            (34, 44),
            (36, 34),
            (32, 42),
            (36, 32),
            (26, 36),
            (24, 28),
            (26, 24),
            (8, 14),
            (16, 20),
            (8, 16),
            (26, 8),
            (10, 26),
            (3, 7),
            (15, 19),
            (3, 15),
            (10, 3),
            (13, 23),
            (10, 13),
            (25, 10),
            (31, 41),
            (27, 31),
            (25, 27),
            (35, 45),
            (37, 47),
            (35, 37),
            (33, 43),
            (35, 33),
            (4, 29),
            (30, 4),
            (35, 30),
            (25, 35)
        ]
    )
}

#[test]
fn stress_test() {
    let mut inputs = vec![];
    for i in 0..1180 {
        inputs.push(i.to_string());
    }
    let size_dict = (0..1180)
        .map(|i| (i.to_string(), 2))
        .collect::<FxHashMap<_, _>>();

    let outputs = vec![];
    let inputs = vec![inputs.clone(), inputs.clone()];
    let contraction_path = cotengra_greedy(&inputs, outputs, size_dict).unwrap();

    assert_eq!(contraction_path, vec![(0, 1)])
}


#[test]
fn test_hyper() {
    // pyo3::prepare_freethreaded_python();
    let inputs = [
        vec![String::from("0")],
        vec![String::from("51")],
        vec![String::from("0"), String::from("2")],
        vec![
            String::from("2"),
            String::from("51"),
            String::from("3"),
            String::from("4"),
        ],
        vec![String::from("3")],
        vec![String::from("4")],
    ];
    let outputs = vec![];

    let size_dict = FxHashMap::from_iter([
        (String::from("51"), 2),
        (String::from("2"), 2),
        (String::from("3"), 2),
        (String::from("4"), 2),
        (String::from("0"), 2),
    ]);

    let contraction_path =
        cotengra_hyperoptimizer(&inputs, outputs, size_dict, "kahypar".to_string(), 150, 15, None).unwrap();

    assert_eq!(
        contraction_path,
        vec![(1, 3), (4, 1), (5, 4), (0, 2), (5, 0)]
    )
}