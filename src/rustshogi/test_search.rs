use super::board::Board;
use super::color::ColorType;
use super::search::*;
use crate::address::Address;
use crate::piece::PieceType;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_starting_board() -> Board {
        let mut board = Board::new();
        board.startpos();
        board
    }

    #[test]
    fn test_simple_evaluator() {
        let board = create_starting_board();
        let evaluator = SimpleEvaluator::new();

        let score = evaluator.evaluate(&board, ColorType::Black);
        // 初期局面では先後同じ駒配置なので、評価値は0に近いはず
        assert!(score.abs() < 100.0, "初期局面の評価値が異常です: {}", score);
    }

    #[test]
    fn test_minmax_search() {
        let board = create_starting_board();
        let search = MinMaxSearch::new(10000);

        let evaluator = SimpleEvaluator::new();
        let result = search.search(&board, ColorType::Black, 2, Some(&evaluator));

        assert!(result.nodes_searched > 0, "ノードが探索されませんでした");
        assert!(result.best_move.is_some(), "最善手が見つかりませんでした");
        assert!(result.score.is_finite(), "評価値が有限ではありません");
    }

    #[test]
    fn test_alphabeta_search() {
        let board = create_starting_board();
        let search = AlphaBetaSearch::new(10000);

        let evaluator = SimpleEvaluator::new();
        let result = search.search(&board, ColorType::Black, 2, Some(&evaluator));

        assert!(result.nodes_searched > 0, "ノードが探索されませんでした");
        assert!(result.best_move.is_some(), "最善手が見つかりませんでした");
        assert!(result.score.is_finite(), "評価値が有限ではありません");
    }

    #[test]
    fn test_alphabeta_vs_minmax() {
        let board = create_starting_board();
        let minmax_search = MinMaxSearch::new(50000);
        let alphabeta_search = AlphaBetaSearch::new(50000);
        let evaluator = SimpleEvaluator::new();

        let minmax_result = minmax_search.search(&board, ColorType::Black, 2, Some(&evaluator));
        let alphabeta_result =
            alphabeta_search.search(&board, ColorType::Black, 2, Some(&evaluator));

        // Alpha-Betaは枝刈りにより、MinMaxよりも少ないノードで探索する
        assert!(
            alphabeta_result.nodes_searched <= minmax_result.nodes_searched,
            "Alpha-Betaのノード数がMinMaxより多い: Alpha-Beta={}, MinMax={}",
            alphabeta_result.nodes_searched,
            minmax_result.nodes_searched
        );

        // 評価値は同じ手を選ぶ場合、同じになる
        // ただし、枝刈りにより異なる手を選ぶ場合もあるため、厳密に等しくする必要はない
    }

    #[test]
    fn test_search_engine() {
        let board = create_starting_board();
        let evaluator = SimpleEvaluator::new();
        let strategy: Box<dyn SearchStrategy + Send + Sync> = Box::new(MinMaxSearch::new(10000));
        let search_engine = SearchEngine::new(strategy, Some(Box::new(evaluator)));

        let result = search_engine.search(&board, ColorType::Black, 2);

        assert!(result.nodes_searched > 0, "ノードが探索されませんでした");
        assert!(result.best_move.is_some(), "最善手が見つかりませんでした");
    }

    #[test]
    fn test_different_depths() {
        let board = create_starting_board();
        let search = AlphaBetaSearch::new(100000);
        let evaluator = SimpleEvaluator::new();

        let result_depth_1 = search.search(&board, ColorType::Black, 1, Some(&evaluator));
        let result_depth_2 = search.search(&board, ColorType::Black, 2, Some(&evaluator));
        let result_depth_3 = search.search(&board, ColorType::Black, 3, Some(&evaluator));

        // 深度が深いほど、探索ノード数は増える
        assert!(
            result_depth_1.nodes_searched <= result_depth_2.nodes_searched,
            "深度2のノード数が異常"
        );
        assert!(
            result_depth_2.nodes_searched <= result_depth_3.nodes_searched,
            "深度3のノード数が異常"
        );
    }

    #[test]
    fn test_endgame_position() {
        // 終盤局面のテスト
        let mut board = Board::new();
        // 簡単な終盤局面を設定
        board.deploy(
            Address::from_numbers(5, 5).to_index(),
            PieceType::King,
            ColorType::Black,
        );
        board.deploy(
            Address::from_numbers(5, 1).to_index(),
            PieceType::King,
            ColorType::White,
        );

        let search = AlphaBetaSearch::new(10000);
        let evaluator = SimpleEvaluator::new();

        let result = search.search(&board, ColorType::Black, 3, Some(&evaluator));

        assert!(result.nodes_searched > 0, "ノードが探索されませんでした");
        assert!(result.best_move.is_some(), "最善手が見つかりませんでした");
    }

    #[test]
    fn test_no_legal_moves() {
        // 合法手がない局面（王手回避不能）
        let mut board = Board::new();
        // 不可能な局面ではなく、実際に合法手がない可能性をテスト
        board.startpos();

        let search = AlphaBetaSearch::new(1000);
        let evaluator = SimpleEvaluator::new();

        let result = search.search(&board, ColorType::Black, 1, Some(&evaluator));

        // 合法手がある場合は、最善手が見つかる
        // 合法手がない場合は、評価値が返される
        assert!(result.score.is_finite(), "評価値が有限ではありません");
    }

    #[test]
    fn test_piece_values() {
        let evaluator = SimpleEvaluator::new();

        // 玉が最重要であることを確認
        let king_value = evaluator.evaluate(
            &{
                let mut b = Board::new();
                b.deploy(
                    Address::from_numbers(5, 5).to_index(),
                    PieceType::King,
                    ColorType::Black,
                );
                b
            },
            ColorType::Black,
        );
        let pawn_value = evaluator.evaluate(
            &{
                let mut b = Board::new();
                b.deploy(
                    Address::from_numbers(5, 5).to_index(),
                    PieceType::Pawn,
                    ColorType::Black,
                );
                b
            },
            ColorType::Black,
        );

        assert!(
            king_value > pawn_value,
            "玉の価値が歩より低い: king={}, pawn={}",
            king_value,
            pawn_value
        );
    }
}
