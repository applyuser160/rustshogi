use rustshogi::common::{alphabet_to_number, char_to_integer, integer_to_char, number_to_alphabet};

#[test]
fn test_common_alphabet_to_number() {
    let num = alphabet_to_number('a');
    assert_eq!(num, '1');
}

#[test]
fn test_common_number_to_alphabet() {
    let ch = number_to_alphabet('2');
    assert_eq!(ch, 'b');
}

#[test]
fn test_common_char_to_integer() {
    let code = char_to_integer('3');
    assert_eq!(code, 3);
}

#[test]
fn test_common_integer_to_char() {
    let ch = integer_to_char(8);
    assert_eq!(ch, '8');
}
