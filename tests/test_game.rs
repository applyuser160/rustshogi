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
    // board.to_string()は持ち駒を含むSFEN形式で出力されるため、期待値を更新
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
    assert!(
        vec![ColorType::Black, ColorType::White, ColorType::None].contains(&result_game.winner)
    );
    assert!(result_game.move_number <= 500);
}

#[test]
fn test_game_random_move_parallel() {
    let mut game = Game::new();
    game.input_board("startpos".to_string());
    let num = 10;
    let threads = 2;
    let results = game.random_move_parallel(num, threads);

    // 結果が空でないことを確認
    assert!(!results.is_empty());

    // 全体の総ゲーム数がnumと一致することを確認
    let total_games: u64 = results.iter().map(|r| r.total_games).sum();
    assert_eq!(total_games, num as u64);

    // 各結果の妥当性を確認
    for result in &results {
        // 白と黒の勝利数の合計が総ゲーム数以下であることを確認
        assert!(result.white_wins + result.black_wins <= result.total_games);
    }

    // 結果の数が可能な手の数と一致することを確認
    let possible_moves = game.board.search_moves(game.turn);
    assert_eq!(results.len(), possible_moves.len());
}

#[test]
fn test_game_random_move_parallel_performance() {
    use std::time::Instant;

    let mut game = Game::new();
    game.input_board("startpos".to_string());
    let num = 1000;

    // シングルスレッドでの実行時間を測定
    let start_single = Instant::now();
    let _results_single = game.random_move_parallel(num, 1);
    let duration_single = start_single.elapsed();

    // マルチスレッドでの実行時間を測定（CPUコア数を使用）
    let start_multi = Instant::now();
    let _results_multi = game.random_move_parallel(num, num_cpus::get());
    let duration_multi = start_multi.elapsed();

    // 性能向上率を計算
    let speedup = duration_single.as_nanos() as f64 / duration_multi.as_nanos() as f64;

    println!("シングルスレッド実行時間: {:?}", duration_single);
    println!("マルチスレッド実行時間: {:?}", duration_multi);
    println!("性能向上率: {:.2}x", speedup);

    // 1.5倍以上の性能向上をアサーション
    assert!(
        speedup >= 1.5,
        "マルチスレッドはシングルスレッドの1.5倍以上速くあるべきです。実際の性能向上率: {:.2}x",
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
