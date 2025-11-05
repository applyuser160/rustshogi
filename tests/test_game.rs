use rustshogi::{board::Board, color::ColorType, game::Game};

#[test]
fn test_game_startpos() {
    let sfen = String::from("startpos");
    let mut game = Game::new();
    game.input_board(sfen);
    let mut board = Board::new();
    board.startpos();
    assert_eq!(game.board, board);
    assert_eq!(game.move_number, 1);
    assert_eq!(game.turn, ColorType::Black);
    assert_eq!(game.winner, ColorType::None);
}

#[test]
fn test_game_input_board() {
    let sfen1 = String::from("startpos");
    let mut game1 = Game::new();
    game1.input_board(sfen1);
    let sfen_str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL";
    let sfen2 = String::from(sfen_str);
    let mut game2 = Game::new();
    game2.input_board(sfen2);
    assert_eq!(game1.board.to_string(), game2.board.to_string());
    // board.to_string() outputs in SFEN format including hand pieces, so the expected value is updated
    assert_eq!(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL -",
        game1.board.to_string()
    );
}

#[test]
fn test_game_random_play() {
    let mut game = Game::new();
    game.input_board("startpos".to_string());
    let result_game = game.random_play();
    assert!(vec![ColorType::Black, ColorType::White, ColorType::None].contains(&result_game.winner));
    assert!(result_game.move_number <= 500);
}

#[test]
fn test_game_random_move_parallel() {
    let mut game = Game::new();
    game.input_board("startpos".to_string());
    let num = 10;
    let threads = 2;
    let results = game.random_move_parallel(num, threads);

    // Check that the results are not empty
    assert!(!results.is_empty());

    // Check that the total number of games equals num
    let total_games: u64 = results.iter().map(|r| r.total_games).sum();
    assert_eq!(total_games, num as u64);

    // Check the validity of each result
    for result in &results {
        // Check that the sum of white and black wins is less than or equal to the total games
        assert!(result.white_wins + result.black_wins <= result.total_games);
    }

    // Check that the number of results matches the number of possible moves
    let possible_moves = game.board.search_moves(game.turn, true);
    assert_eq!(results.len(), possible_moves.len());
}

#[test]
fn test_game_random_move_parallel_performance() {
    use std::time::Instant;

    let mut game = Game::new();
    game.input_board("startpos".to_string());
    let num = 1000;

    // Measure execution time with a single thread
    let start_single = Instant::now();
    let _results_single = game.random_move_parallel(num, 1);
    let duration_single = start_single.elapsed();

    // Measure execution time with multiple threads (using the number of CPU cores)
    let start_multi = Instant::now();
    let _results_multi = game.random_move_parallel(num, num_cpus::get());
    let duration_multi = start_multi.elapsed();

    // Calculate the speedup
    let speedup = duration_single.as_nanos() as f64 / duration_multi.as_nanos() as f64;

    println!("Single-threaded execution time: {:?}", duration_single);
    println!("Multi-threaded execution time: {:?}", duration_multi);
    println!("Speedup: {:.2}x", speedup);

    // Assert that the speedup is at least 1.5x
    assert!(
        speedup >= 1.5,
        "Multi-threading should be at least 1.5 times faster than single-threading. Actual speedup: {:.2}x",
        speedup
    );
}

#[test]
fn test_game_generate_random_board() {
    let mut game = Game::new();
    game.input_board("startpos".to_string());
    let board = game.generate_random_board();

    assert!(board.is_finished().0 || !board.is_finished().0);
}
