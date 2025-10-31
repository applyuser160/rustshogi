use rustshogi::{
    color::ColorType,
    piece::{MoveType, Piece, PieceType},
};

#[test]
fn test_piece_from_usize() {
    let piece_type = PieceType::from_usize(1);
    assert_eq!(piece_type as u8, 1);
}

#[test]
fn test_piece_to_string() {
    let piece_type = PieceType::from_usize(1);
    assert_eq!(piece_type.to_string(), String::from('k'));
}

#[test]
fn test_piece_new() {
    let piece = Piece::new();
    assert_eq!(piece.owner, ColorType::None);
    assert_eq!(piece.piece_type, PieceType::None);
}

#[test]
fn test_piece_from() {
    let piece = Piece::from(ColorType::Black, PieceType::Dragon);
    assert_eq!(piece.owner, ColorType::Black);
    assert_eq!(piece.piece_type, PieceType::Dragon);
}

#[test]
fn test_piece_from_u8_1() {
    /* 0d142 = 0b01001110 */
    let piece = Piece::from_u8(78);
    assert_eq!(piece.owner, ColorType::White);
    assert_eq!(piece.piece_type, PieceType::ProPawn);
}

#[test]
fn test_piece_from_u8_2() {
    /* 0d142 = 0b11111111 */
    let piece = Piece::from_u8(255);
    assert_eq!(piece.owner, ColorType::White);
    assert_eq!(piece.piece_type, PieceType::None);
}

#[test]
fn test_piece_from_char_1() {
    let piece = Piece::from_char('K');
    assert_eq!(piece.owner, ColorType::Black);
    assert_eq!(piece.piece_type, PieceType::King);
}

#[test]
fn test_piece_from_char_2() {
    let piece = Piece::from_char('p');
    assert_eq!(piece.owner, ColorType::White);
    assert_eq!(piece.piece_type, PieceType::Pawn);
}

#[test]
fn test_piece_from_string_1() {
    let piece = Piece::from_string("+p");
    assert_eq!(piece.owner, ColorType::White);
    assert_eq!(piece.piece_type, PieceType::ProPawn);
}

#[test]
fn test_piece_from_string_2() {
    let piece = Piece::from_string("B");
    assert_eq!(piece.owner, ColorType::Black);
    assert_eq!(piece.piece_type, PieceType::Bichop);
}

#[test]
fn test_piece_to_u8_and_to_string() {
    let piece = Piece::from(ColorType::Black, PieceType::Dragon);
    assert_eq!(piece.to_u8(), 9);
    assert_eq!(piece.to_string(), String::from("+R"));
}

#[test]
fn test_piece_get_movetype_1() {
    let result = Piece::get_movetype(PieceType::King);
    assert_eq!(result[0], MoveType::Short);
    assert_eq!(result[1], MoveType::Short);
    assert_eq!(result[2], MoveType::Short);
    assert_eq!(result[3], MoveType::Short);
    assert_eq!(result[4], MoveType::Short);
    assert_eq!(result[5], MoveType::Short);
    assert_eq!(result[6], MoveType::Short);
    assert_eq!(result[7], MoveType::Short);
}

#[test]
fn test_piece_get_movetype_2() {
    let result = Piece::get_movetype(PieceType::Bichop);
    assert_eq!(result[0], MoveType::None);
    assert_eq!(result[1], MoveType::Long);
    assert_eq!(result[2], MoveType::None);
    assert_eq!(result[3], MoveType::Long);
    assert_eq!(result[4], MoveType::None);
    assert_eq!(result[5], MoveType::Long);
    assert_eq!(result[6], MoveType::None);
    assert_eq!(result[7], MoveType::Long);
}

#[test]
fn test_piece_able_pro_1() {
    let result = Piece::able_pro(PieceType::Gold);
    assert_eq!(result, false);
}

#[test]
fn test_piece_able_pro_2() {
    let result = Piece::able_pro(PieceType::Bichop);
    assert_eq!(result, true);
}
