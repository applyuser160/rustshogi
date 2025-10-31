use rustshogi::bitboard::{generate_column, BitBoard};

#[test]
fn test_bitboard_new() {
    let bitboard = BitBoard::new();
    assert_eq!((bitboard.to_u128() >> (127 - 0)) & 1, 0);
}

#[test]
fn test_bitboard_from_u128() {
    let bitboard = BitBoard::from_u128(548949983232);
    assert_eq!((bitboard.to_u128() >> (127 - 88)) & 1, 0);
    assert_eq!((bitboard.to_u128() >> (127 - 89)) & 1, 1);
    assert_eq!((bitboard.to_u128() >> (127 - 90)) & 1, 1);
}

#[test]
fn test_bitboard_from_str() {
    let s = "
        00000000000
        00000000000
        00000000000
        00000000000
        00000000000
        00000000000
        00000000000
        00000000000
        00000000000
        00000000000
        00000000000";
    let bitboard = BitBoard::from_str(s);
    assert_eq!((bitboard.to_u128() >> (127 - 0)) & 1, 0);
}

#[test]
fn test_bitboard_to_u128() {
    let bitboard = BitBoard::from_u128(548949983232);
    assert_eq!(bitboard.to_u128(), 548949983232);
}

#[test]
fn test_bitboard_set_bit() {
    let mut bitboard = BitBoard::new();
    // set_true相当: ビットを設定するために直接ビット演算を使用
    let bit_mask = BitBoard::from_u128(1u128 << (127 - 89));
    bitboard |= bit_mask;
    assert_eq!((bitboard.to_u128() >> (127 - 89)) & 1, 1);
}

#[test]
fn test_bitboard_unset_bit() {
    let mut bitboard = BitBoard::from_u128(548949983232);
    // set_false相当: ビットをクリアするために直接ビット演算を使用
    let bit_mask = BitBoard::from_u128(1u128 << (127 - 89));
    let all_bits = BitBoard::from_u128(u128::MAX);
    let not_mask = all_bits ^ bit_mask;
    bitboard &= not_mask;
    assert_eq!((bitboard.to_u128() >> (127 - 89)) & 1, 0);
}

#[test]
fn test_bitboard_get_trues() {
    let bitboard = BitBoard::from_u128(548949983232);
    let trues = bitboard.get_trues();
    assert_eq!(trues.len(), 2);
    assert_eq!(trues[0], 89);
    assert_eq!(trues[1], 90);
}

#[test]
fn test_bitboard_and() {
    let bitboard1 = BitBoard::from_u128(548949983232);
    let bitboard2 = BitBoard::from_u128(1097899966464);
    let result = bitboard1 & bitboard2;
    assert_eq!(result.to_u128(), 548949983232);
}

#[test]
fn test_bitboard_or() {
    let bitboard1 = BitBoard::from_u128(548949983232);
    let bitboard2 = BitBoard::from_u128(1097899966464);
    let result = bitboard1 | bitboard2;
    assert_eq!(result.to_u128(), 1646849949696);
}

#[test]
fn test_bitboard_xor() {
    let bitboard1 = BitBoard::from_u128(548949983232);
    let bitboard2 = BitBoard::from_u128(1097899966464);
    let result = bitboard1 ^ bitboard2;
    assert_eq!(result.to_u128(), 1097899966464);
}

#[test]
fn test_bitboard_shift() {
    let bitboard = BitBoard::from_u128(548949983232);
    let result = bitboard >> 1;
    assert_eq!((result.to_u128() >> (127 - 88)) & 1, 1);
    assert_eq!((result.to_u128() >> (127 - 89)) & 1, 1);
}

#[test]
fn test_bitboard_generate_column() {
    let column = generate_column(5);
    let trues = column.get_trues();
    assert_eq!(trues.len(), 11);
    assert_eq!(trues[0], 5);
    assert_eq!(trues[10], 115);
}
