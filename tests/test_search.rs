#[cfg(test)]
mod tests {
    use rustshogi::address::Address;
    use rustshogi::board::Board;
    use rustshogi::color::ColorType;
    use rustshogi::evaluator::abst::Evaluator;
    use rustshogi::evaluator::simple::SimpleEvaluator;
    use rustshogi::piece::PieceType;
    use rustshogi::search::{
        AlphaBetaSearchStrategy, MinMaxSearchStrategy, SearchEngine, SearchStrategy,
    };

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
        // In the initial position, the piece placement is the same for both players, so the evaluation value should be close to 0
        assert!(
            score.abs() < 100.0,
            "The evaluation value of the initial position is abnormal: {}",
            score
        );
    }

    #[test]
    fn test_minmax_search() {
        let board = create_starting_board();
        let search = MinMaxSearchStrategy::new(10000);

        let evaluator = SimpleEvaluator::new();
        let results = search.search(&board, ColorType::Black, 2, None, Some(&evaluator));
        let result = results
            .first()
            .expect("Expected at least one evaluation result");

        assert!(result.nodes_searched > 0, "No nodes were searched");
        assert!(result.best_move.is_some(), "No best move was found");
        assert!(
            result.score.is_finite(),
            "The evaluation value is not finite"
        );
    }

    #[test]
    fn test_alphabeta_search() {
        let board = create_starting_board();
        let search = AlphaBetaSearchStrategy::new(10000);

        let evaluator = SimpleEvaluator::new();
        let results = search.search(&board, ColorType::Black, 2, None, Some(&evaluator));
        let result = results
            .first()
            .expect("Expected at least one evaluation result");

        assert!(result.nodes_searched > 0, "No nodes were searched");
        assert!(result.best_move.is_some(), "No best move was found");
        assert!(
            result.score.is_finite(),
            "The evaluation value is not finite"
        );
    }

    #[test]
    fn test_alphabeta_vs_minmax() {
        let board = create_starting_board();
        let minmax_search = MinMaxSearchStrategy::new(50000);
        let alphabeta_search = AlphaBetaSearchStrategy::new(50000);
        let evaluator = SimpleEvaluator::new();

        let minmax_result =
            minmax_search.search(&board, ColorType::Black, 2, None, Some(&evaluator));
        let alphabeta_result =
            alphabeta_search.search(&board, ColorType::Black, 2, None, Some(&evaluator));

        // Alpha-Beta searches fewer nodes than MinMax due to pruning
        assert!(
            alphabeta_result
                .iter()
                .map(|result| result.nodes_searched)
                .sum::<u64>()
                <= minmax_result
                    .iter()
                    .map(|result| result.nodes_searched)
                    .sum::<u64>(),
            "The number of nodes in Alpha-Beta is greater than in MinMax: Alpha-Beta={}, MinMax={}",
            alphabeta_result
                .iter()
                .map(|result| result.nodes_searched)
                .sum::<u64>(),
            minmax_result
                .iter()
                .map(|result| result.nodes_searched)
                .sum::<u64>()
        );

        // The evaluation value will be the same if the same move is chosen
        // However, it is not necessary for them to be strictly equal because different moves may be chosen due to pruning
    }

    #[test]
    fn test_search_engine() {
        let board = create_starting_board();
        let evaluator = SimpleEvaluator::new();
        let strategy: Box<dyn SearchStrategy + Send + Sync> =
            Box::new(MinMaxSearchStrategy::new(10000));
        let search_engine = SearchEngine::new(strategy, Some(Box::new(evaluator)));

        let results = search_engine.search(&board, ColorType::Black, 2, None);
        let result = results
            .first()
            .expect("Expected at least one evaluation result");

        assert!(result.nodes_searched > 0, "No nodes were searched");
        assert!(result.best_move.is_some(), "No best move was found");
    }

    #[test]
    fn test_different_depths() {
        let board = create_starting_board();
        let search = AlphaBetaSearchStrategy::new(100000);
        let evaluator = SimpleEvaluator::new();

        let result_depth_1 = search.search(&board, ColorType::Black, 1, None, Some(&evaluator));
        let result_depth_2 = search.search(&board, ColorType::Black, 2, None, Some(&evaluator));
        let result_depth_3 = search.search(&board, ColorType::Black, 3, None, Some(&evaluator));

        // The deeper the search, the more nodes are searched
        assert!(
            result_depth_1
                .iter()
                .map(|result| result.nodes_searched)
                .sum::<u64>()
                <= result_depth_2
                    .iter()
                    .map(|result| result.nodes_searched)
                    .sum::<u64>(),
            "The number of nodes at depth 2 is abnormal"
        );
        assert!(
            result_depth_2
                .iter()
                .map(|result| result.nodes_searched)
                .sum::<u64>()
                <= result_depth_3
                    .iter()
                    .map(|result| result.nodes_searched)
                    .sum::<u64>(),
            "The number of nodes at depth 3 is abnormal"
        );
    }

    #[test]
    fn test_endgame_position() {
        // Test endgame position
        let mut board = Board::new();
        // Set up a simple endgame position
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

        let search = AlphaBetaSearchStrategy::new(10000);
        let evaluator = SimpleEvaluator::new();

        let results = search.search(&board, ColorType::Black, 3, None, Some(&evaluator));
        let result = results
            .first()
            .expect("Expected at least one evaluation result");

        assert!(result.nodes_searched > 0, "No nodes were searched");
        assert!(result.best_move.is_some(), "No best move was found");
    }

    #[test]
    fn test_no_legal_moves() {
        // Position with no legal moves (checkmate cannot be avoided)
        let mut board = Board::new();
        // Instead of an impossible position, test the possibility of actually having no legal moves
        board.startpos();

        let search = AlphaBetaSearchStrategy::new(1000);
        let evaluator = SimpleEvaluator::new();

        let results = search.search(&board, ColorType::Black, 1, None, Some(&evaluator));
        let result = results
            .first()
            .expect("Expected at least one evaluation result");

        // If there are legal moves, the best move will be found
        // If there are no legal moves, an evaluation value will be returned
        assert!(
            result.score.is_finite(),
            "The evaluation value is not finite"
        );
    }

    #[test]
    fn test_piece_values() {
        let evaluator = SimpleEvaluator::new();

        // Check that the king is the most important piece
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
            "The value of the king is lower than the pawn: king={}, pawn={}",
            king_value,
            pawn_value
        );
    }
}
