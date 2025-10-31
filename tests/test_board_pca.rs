use rustshogi::bitboard::BitBoard;
use rustshogi::board::Board;
use rustshogi::color::ColorType;
use rustshogi::pca::{learn_simple_pca, set_global_pca_transform};
use rustshogi::piece::PieceType;

#[test]
fn test_board_new() {
    let board = Board::new();
    assert_eq!(board.has_piece, BitBoard::new());
    assert_eq!(board.player_prossesion[0], BitBoard::new());
    assert_eq!(board.player_prossesion[1], BitBoard::new());
}

#[test]
fn test_board_to_vector_no_compression() {
    let board = Board::new();
    let vector = board.to_vector(None);

    // 2304 (盤面) + 16 (持ち駒) = 2320次元
    assert_eq!(vector.len(), 2320);
}

#[test]
fn test_board_to_vector_with_compression() {
    let board = Board::new();
    let vector = board.to_vector(Some(100));

    assert_eq!(vector.len(), 100);
}

#[test]
fn test_board_to_vector_with_pca() {
    let board = Board::new();

    // サンプルデータでPCAを学習
    let samples = vec![vec![1.0; 2320], vec![2.0; 2320], vec![3.0; 2320]];
    let pca_transform = learn_simple_pca(&samples, 50);
    set_global_pca_transform(pca_transform);

    let vector = board.to_vector(Some(50));
    assert_eq!(vector.len(), 50);
}

#[test]
fn test_board_startpos() {
    let mut board = Board::new();
    board.startpos();

    // 開始局面では駒が配置されているはず
    assert_ne!(board.has_piece, BitBoard::new());
}

#[test]
fn test_board_deploy() {
    let mut board = Board::new();
    board.deploy(0, PieceType::King, ColorType::Black);

    // 駒が配置されているはず
    assert_ne!(board.has_piece, BitBoard::new());
}

#[test]
fn test_board_get_piece() {
    let mut board = Board::new();
    board.deploy(0, PieceType::King, ColorType::Black);

    let piece = board.get_piece(0);
    assert_eq!(piece.piece_type, PieceType::King);
    assert_eq!(piece.owner, ColorType::Black);
}

#[test]
fn test_board_is_finished() {
    let mut board = Board::new();
    board.startpos(); // 開始局面を設定
    let (is_finished, winner) = board.is_finished();

    // 開始局面では終了していない
    assert!(!is_finished);
    assert_eq!(winner, ColorType::None);
}
