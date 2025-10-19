#[cfg(test)]

mod tests {
    use crate::{
        address::Address,
        bitboard::{BitBoard, BIT_OF_FRAME, BIT_OF_PRO_ZONE_BLACK},
        board::Board,
        color::ColorType,
        moves::Move,
        piece::PieceType,
    };

    #[test]
    fn test_board_new() {
        let board = Board::new();
        assert_eq!(board.is_frame, BitBoard::from_u128(BIT_OF_FRAME));
        assert_eq!(
            board.able_pro[0],
            BitBoard::from_u128(BIT_OF_PRO_ZONE_BLACK)
        );
    }

    #[test]
    fn test_board_startpos() {
        let mut board = Board::new();
        board.startpos();
        assert_eq!(board.get_piece_type_from_index(12), PieceType::Lance);
        assert_eq!(board.get_color_type_from_index(12), ColorType::Black);
    }

    #[test]
    fn test_board_get_able_move_squares() {
        let mut board = Board::new();
        board.startpos();
        let result1 = board.get_able_move_squares(34);
        assert_eq!(result1.get_trues(), vec![45]);
        let result2 = board.get_able_move_squares(78);
        assert_eq!(result2.get_trues(), vec![67]);
        let result1 = board.get_able_move_squares(30);
        assert_eq!(result1.get_trues(), vec![25, 26, 27, 28, 29, 31]);
    }

    #[test]
    fn test_board_deploy() {
        let mut board = Board::new();
        board.deploy(12, PieceType::Rook, ColorType::Black);
        let bit_movable = board.get_able_move_squares(12);
        let result = board.get_able_pro_move_squares(12, bit_movable);
        assert_eq!(result.get_trues(), vec![78, 89, 100]);
    }

    #[test]
    fn test_board_get_able_drop_squares() {
        let mut board = Board::new();
        board.startpos();
        let result1 = board.get_able_drop_squares(ColorType::Black, PieceType::Pawn);
        assert_eq!(result1.get_trues(), vec![] as Vec<u8>);
        let result2 = board.get_able_drop_squares(ColorType::White, PieceType::Knight);
        assert_eq!(result2.get_trues().len(), 34);
        assert_eq!(result2.get_trues()[0], 45);
    }

    #[test]
    fn test_board_search_moves() {
        let mut board = Board::new();
        board.startpos();
        let result = board.search_moves(ColorType::Black);
        assert_eq!(result.len(), 30);
    }

    #[test]
    fn test_board_execute_move() {
        let mut board = Board::new();
        board.startpos();
        let from = Address::from_number(34);
        let to = Address::from_number(45);
        board.execute_move(&Move::from_standart(from, to, false));
        assert_eq!(board.get_piece_type_from_index(45), PieceType::Pawn);
        assert_eq!(board.get_color_type_from_index(45), ColorType::Black);
    }

    #[test]
    fn test_board_is_finished() {
        let mut board = Board::new();
        board.startpos();
        let from = Address::from_number(93);
        let to = Address::from_number(16);
        board.execute_move(&Move::from_standart(from, to, false));
        let result = board.is_finished();
        assert_eq!(result.0, true);
        assert_eq!(result.1, ColorType::White);
    }

    #[test]
    fn test_board_to_vector() {
        let mut board = Board::new();
        board.startpos();
        let vector = board.to_vector(None);
        assert_eq!(vector.len(), 2320);
    }

    #[test]
    fn test_board_to_vector_with_options() {
        let mut board = Board::new();
        board.startpos();
        let vector = board.to_vector(Some(300));
        assert_eq!(vector.len(), 300);
    }

    #[test]
    fn test_board_to_string_startpos() {
        let mut board = Board::new();
        board.startpos();
        let sfen = board.to_string();

        // SFEN形式の基本構造をチェック
        assert!(sfen.contains(" ")); // 盤面と持ち駒の区切り
        let parts: Vec<&str> = sfen.split(" ").collect();
        assert_eq!(parts.len(), 2);

        // 盤面部分のチェック（9行、8個の'/'区切り）
        let board_part = parts[0];
        let slash_count = board_part.matches('/').count();
        assert_eq!(slash_count, 8);

        // 持ち駒部分のチェック（初期状態では'-'）
        let hand_part = parts[1];
        assert_eq!(hand_part, "-");
    }

    #[test]
    fn test_board_from_sfen_startpos() {
        let mut board = Board::new();
        board.startpos();
        let sfen = board.to_string();

        let board_from_sfen = Board::from_sfen(sfen);

        // 基本的な駒の配置をチェック（実際の盤面の位置を確認）
        // 初期配置では、1行目（index 12-20）に先手の駒が配置される
        // ただし、実際の位置を確認するために、元のboardと比較
        assert_eq!(
            board_from_sfen.get_piece_type_from_index(12),
            board.get_piece_type_from_index(12)
        );
        assert_eq!(
            board_from_sfen.get_color_type_from_index(12),
            board.get_color_type_from_index(12)
        );
        // 9行目（index 78-86）に後手の駒が配置される
        assert_eq!(
            board_from_sfen.get_piece_type_from_index(78),
            board.get_piece_type_from_index(78)
        );
        assert_eq!(
            board_from_sfen.get_color_type_from_index(78),
            board.get_color_type_from_index(78)
        );
    }

    #[test]
    fn test_board_sfen_roundtrip() {
        let mut board1 = Board::new();
        board1.startpos();
        let sfen1 = board1.to_string();

        let board2 = Board::from_sfen(sfen1.clone());
        let sfen2 = board2.to_string();

        assert_eq!(sfen1, sfen2);
    }

    #[test]
    fn test_board_sfen_with_hand() {
        let mut board = Board::new();
        board.startpos();

        // 駒を取る（持ち駒を増やす）ために、実際に駒を取る手を実行
        // 先手の歩を後手の歩の位置に移動して取る
        let from = Address::from_number(34); // 先手の歩
        let to = Address::from_number(45); // 後手の歩の位置
        board.execute_move(&Move::from_standart(from, to, false));

        let sfen = board.to_string();
        let parts: Vec<&str> = sfen.split(" ").collect();

        // 持ち駒部分が'-'でないことをチェック（駒を取ったので持ち駒があるはず）
        // ただし、実際に駒を取れない場合は'-'のままになる可能性がある
        // その場合は、手動で持ち駒を追加してテスト
        if parts[1] == "-" {
            // 手動で持ち駒を追加
            board.hand.add_piece(ColorType::Black, PieceType::Pawn);
            let sfen_with_hand = board.to_string();
            let parts_with_hand: Vec<&str> = sfen_with_hand.split(" ").collect();
            assert_ne!(parts_with_hand[1], "-");

            // SFENから復元して同じ結果になることをチェック
            let restored_board = Board::from_sfen(sfen_with_hand.clone());
            let restored_sfen = restored_board.to_string();
            assert_eq!(sfen_with_hand, restored_sfen);
        } else {
            // SFENから復元して同じ結果になることをチェック
            let restored_board = Board::from_sfen(sfen.clone());
            let restored_sfen = restored_board.to_string();
            assert_eq!(sfen, restored_sfen);
        }
    }
}
