use super::board::Board;
use super::color::{convert_from_string, get_reverse_color, ColorType};
use super::mctsresult::MctsResult;
use super::moves::Move;
use super::piece::Piece;
use super::random::Random;
use num_cpus;
use rayon::prelude::*;

use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Game {
    #[pyo3(get, set)]
    pub board: Board,
    #[pyo3(get, set)]
    pub move_number: u16,
    #[pyo3(get, set)]
    pub turn: ColorType,
    #[pyo3(get, set)]
    pub winner: ColorType,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            move_number: 1,
            turn: ColorType::Black,
            winner: ColorType::None,
        }
    }

    pub fn from(board: Board, move_number: u16, turn: ColorType, winner: ColorType) -> Self {
        Self {
            board,
            move_number,
            turn,
            winner,
        }
    }

    pub fn input_board(&mut self, sfen: String) {
        self.board = Board::from_sfen(sfen);
    }

    pub fn input_hand(&mut self, sfen: String) {
        if sfen == "-" {
            return;
        }
        let mut current_sfen = sfen.chars();
        while let Some(ch) = current_sfen.next() {
            if ch.is_ascii_digit() {
                let consecutive = ch.to_digit(10).unwrap() as u8;
                let piece = Piece::from_char(current_sfen.next().unwrap());
                self.board
                    .hand
                    .add_pieces(piece.owner, piece.piece_type, consecutive);
            } else {
                let piece = Piece::from_char(ch);
                self.board.hand.add_piece(piece.owner, piece.piece_type);
            }
        }
    }

    pub fn input_move_number(&mut self, sfen: String) {
        self.move_number = sfen.parse::<u16>().unwrap_or(0);
    }

    pub fn input_turn(&mut self, sfen: String) {
        self.turn = convert_from_string(sfen.chars().next().unwrap_or(' '));
    }

    pub fn is_finished(&self) -> (bool, ColorType) {
        if self.move_number >= 500 {
            (true, ColorType::None)
        } else {
            self.board.is_finished()
        }
    }

    pub fn execute_move(&mut self, mv: &Move) {
        self.board.execute_move(mv);
        self.move_number += 1;
        self.turn = get_reverse_color(self.turn);
    }

    pub fn one_play(&mut self) -> Self {
        // used for benchmark only
        while !self.is_finished().0 {
            let moves = self.board.search_moves(self.turn, true);
            let amove = &moves[0];
            self.execute_move(amove);
            let is_finish = self.is_finished();
            if is_finish.0 {
                self.winner = is_finish.1;
                break;
            }
        }
        self.clone()
    }

    pub fn random_play(&mut self) -> Self {
        while !self.is_finished().0 {
            let moves = self.board.search_moves(self.turn, true);
            if moves.is_empty() {
                break;
            }

            let mut random = Random::new(0, (moves.len() - 1) as u16);
            let amove = &moves[random.generate_one() as usize];
            self.execute_move(amove);
            let is_finish = self.is_finished();
            if is_finish.0 {
                self.winner = is_finish.1;
                break;
            }
        }
        self.clone()
    }

    pub fn random_move_parallel(&self, num: usize, num_threads: usize) -> Vec<MctsResult> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let next_moves = self.board.search_moves(self.turn, true);
        let next_move_count = next_moves.len();

        if next_move_count == 0 {
            return vec![];
        }

        // Count how many simulations to run for each move.
        let mut counts = vec![0; next_move_count];
        let mut random_gen = Random::new(0, (next_move_count - 1) as u16);
        for _ in 0..num {
            counts[random_gen.generate_one() as usize] += 1;
        }

        // Initialize MctsResult for each move.
        let mut results: Vec<MctsResult> = next_moves
            .iter()
            .map(|mv| MctsResult::from(self.board.clone(), mv.clone()))
            .collect();

        // Run simulations in parallel.
        let simulation_results: Vec<(ColorType, usize)> = pool.install(|| {
            counts
                .into_par_iter()
                .enumerate()
                .flat_map(|(move_index, num_sims_for_this_move)| {
                    if num_sims_for_this_move == 0 {
                        return Vec::new();
                    }

                    // Clone and advance the game state ONCE for this move.
                    let mut initial_game_clone = self.clone();
                    initial_game_clone.execute_move(&next_moves[move_index]);

                    // Run simulations for this move sequentially within this parallel task.
                    (0..num_sims_for_this_move)
                        .map(|_| {
                            let mut game_clone = initial_game_clone.clone();

                            // Perform a random playout.
                            while !game_clone.is_finished().0 {
                                let moves = game_clone.board.search_moves(game_clone.turn, false);
                                if moves.is_empty() {
                                    break;
                                }
                                let move_count = moves.len();
                                let mut random = Random::new(0, (move_count - 1) as u16);
                                let random_move = &moves[random.generate_one() as usize];
                                game_clone.execute_move(random_move);
                            }

                            let (_is_finished, winner) = game_clone.is_finished();
                            (winner, move_index)
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        });

        // 結果を各MctsResultに集計
        for (winner, move_index) in simulation_results {
            results[move_index].plus_result(winner);
        }

        results
    }

    pub fn random_move_sequential(&self, num: usize) -> Vec<MctsResult> {
        let next_moves = self.board.search_moves(self.turn, true);
        let next_move_count = next_moves.len();

        if next_move_count == 0 {
            return vec![];
        }

        // 各手に対してMctsResultを初期化
        let mut results: Vec<MctsResult> = next_moves
            .iter()
            .map(|mv| MctsResult::from(self.board.clone(), mv.clone()))
            .collect();

        // 順次処理でシミュレーションを実行
        for _i in 0..num {
            // ランダムに手を選択
            let mut random = Random::new(0, (next_move_count - 1) as u16);
            let selected_move_index = random.generate_one() as usize;

            // 選択された手でゲームを開始
            let mut game_clone = self.clone();
            game_clone.execute_move(&next_moves[selected_move_index]);

            // ランダムプレイでゲーム終了まで実行
            while !game_clone.is_finished().0 {
                let moves = game_clone.board.search_moves(game_clone.turn, true);
                if moves.is_empty() {
                    break;
                }
                let move_count = moves.len();
                let mut random = Random::new(0, (move_count - 1) as u16);
                let random_move = &moves[random.generate_one() as usize];
                game_clone.execute_move(random_move);
            }

            let (_is_finished, winner) = game_clone.is_finished();
            results[selected_move_index].plus_result(winner);
        }

        results
    }

    pub fn random_move_parallel_chunked(&self, num: usize, num_threads: usize) -> Vec<MctsResult> {
        let next_moves = self.board.search_moves(self.turn, true);
        let next_move_count = next_moves.len();

        if next_move_count == 0 {
            return vec![];
        }

        // 各手に対してMctsResultを初期化
        let mut results: Vec<MctsResult> = next_moves
            .iter()
            .map(|mv| MctsResult::from(self.board.clone(), mv.clone()))
            .collect();

        // チャンクサイズを計算（各スレッドに均等に分散）
        let chunk_size = num.div_ceil(num_threads);

        // スレッドプールを設定
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        // 各スレッドでチャンクを処理
        let chunk_results: Vec<Vec<(ColorType, usize)>> = pool.install(|| {
            (0..num_threads)
                .into_par_iter()
                .map(|thread_id| {
                    let start = thread_id * chunk_size;
                    let end = std::cmp::min(start + chunk_size, num);
                    let mut thread_results = Vec::new();

                    for _i in start..end {
                        // ランダムに手を選択
                        let mut random = Random::new(0, (next_move_count - 1) as u16);
                        let selected_move_index = random.generate_one() as usize;

                        // 選択された手でゲームを開始
                        let mut game_clone = self.clone();
                        game_clone.execute_move(&next_moves[selected_move_index]);

                        // ランダムプレイでゲーム終了まで実行
                        while !game_clone.is_finished().0 {
                            let moves = game_clone.board.search_moves(game_clone.turn, true);
                            if moves.is_empty() {
                                break;
                            }
                            let move_count = moves.len();
                            let mut random = Random::new(0, (move_count - 1) as u16);
                            let random_move = &moves[random.generate_one() as usize];
                            game_clone.execute_move(random_move);
                        }

                        let (_is_finished, winner) = game_clone.is_finished();
                        thread_results.push((winner, selected_move_index));
                    }

                    thread_results
                })
                .collect()
        });

        // 結果を統合
        for thread_result in chunk_results {
            for (winner, move_index) in thread_result {
                results[move_index].plus_result(winner);
            }
        }

        results
    }

    pub fn random_move_parallel_thread_local(
        &self,
        num: usize,
        num_threads: usize,
    ) -> Vec<MctsResult> {
        let next_moves = self.board.search_moves(self.turn, true);
        let next_move_count = next_moves.len();

        if next_move_count == 0 {
            return vec![];
        }

        // スレッドプールを設定
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        // スレッドローカルで結果を集計
        let results: Vec<MctsResult> = pool.install(|| {
            use std::sync::Mutex;
            let results = Mutex::new(vec![
                MctsResult::from(
                    self.board.clone(),
                    next_moves[0].clone()
                );
                next_move_count
            ]);

            (0..num).into_par_iter().for_each(|_| {
                // ランダムに手を選択
                let mut random = Random::new(0, (next_move_count - 1) as u16);
                let selected_move_index = random.generate_one() as usize;

                // 選択された手でゲームを開始
                let mut game_clone = self.clone();
                game_clone.execute_move(&next_moves[selected_move_index]);

                // ランダムプレイでゲーム終了まで実行
                while !game_clone.is_finished().0 {
                    let moves = game_clone.board.search_moves(game_clone.turn, true);
                    if moves.is_empty() {
                        break;
                    }
                    let move_count = moves.len();
                    let mut random = Random::new(0, (move_count - 1) as u16);
                    let random_move = &moves[random.generate_one() as usize];
                    game_clone.execute_move(random_move);
                }

                let (_is_finished, winner) = game_clone.is_finished();

                // 結果をスレッドセーフに更新
                if let Ok(mut results_guard) = results.lock() {
                    results_guard[selected_move_index].plus_result(winner);
                }
            });

            results.into_inner().unwrap()
        });

        results
    }

    pub fn random_move_parallel_batched(&self, num: usize, num_threads: usize) -> Vec<MctsResult> {
        let next_moves = self.board.search_moves(self.turn, true);
        let next_move_count = next_moves.len();

        if next_move_count == 0 {
            return vec![];
        }

        // 各手に対してMctsResultを初期化
        let mut results: Vec<MctsResult> = next_moves
            .iter()
            .map(|mv| MctsResult::from(self.board.clone(), mv.clone()))
            .collect();

        // バッチサイズを計算（各スレッドが処理するバッチ数）
        let batch_size = std::cmp::max(1, num / (num_threads * 4)); // 各スレッドに4つのバッチを割り当て

        // スレッドプールを設定
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        // バッチ単位で並列処理
        let batch_results: Vec<Vec<(ColorType, usize)>> = pool.install(|| {
            let num_batches = num.div_ceil(batch_size);
            (0..num_batches)
                .into_par_iter()
                .map(|batch_id| {
                    let batch_start = batch_id * batch_size;
                    let batch_end = std::cmp::min(batch_start + batch_size, num);
                    let mut batch_results = Vec::new();

                    for _i in batch_start..batch_end {
                        // ランダムに手を選択
                        let mut random = Random::new(0, (next_move_count - 1) as u16);
                        let selected_move_index = random.generate_one() as usize;

                        // 選択された手でゲームを開始
                        let mut game_clone = self.clone();
                        game_clone.execute_move(&next_moves[selected_move_index]);

                        // ランダムプレイでゲーム終了まで実行
                        while !game_clone.is_finished().0 {
                            let moves = game_clone.board.search_moves(game_clone.turn, true);
                            if moves.is_empty() {
                                break;
                            }
                            let move_count = moves.len();
                            let mut random = Random::new(0, (move_count - 1) as u16);
                            let random_move = &moves[random.generate_one() as usize];
                            game_clone.execute_move(random_move);
                        }

                        let (_is_finished, winner) = game_clone.is_finished();
                        batch_results.push((winner, selected_move_index));
                    }

                    batch_results
                })
                .collect()
        });

        // 結果を統合
        for batch_result in batch_results {
            for (winner, move_index) in batch_result {
                results[move_index].plus_result(winner);
            }
        }

        results
    }

    pub fn generate_random_board(&mut self) -> Board {
        let mut random = Random::new(0, 150);
        let move_count = random.generate_one() as usize;

        for _ in 0..move_count {
            let moves = self.board.search_moves(self.turn, true);
            if moves.is_empty() {
                break;
            }

            let mut random = Random::new(0, (moves.len() - 1) as u16);
            let amove = &moves[random.generate_one() as usize].clone();
            self.execute_move(amove);

            if self.is_finished().0 {
                break;
            }
        }
        self.board.clone()
    }
}

#[pymethods]
impl Game {
    #[pyo3(name = "random_move")]
    #[pyo3(signature = (num, threads = None))]
    pub fn python_random_move(&self, num: usize, threads: Option<usize>) -> Vec<MctsResult> {
        let num_threads = threads.unwrap_or_else(num_cpus::get);
        self.random_move_parallel(num, num_threads)
    }

    #[new]
    #[pyo3(signature = (board = Board::new_for_python("startpos".to_string()), move_number = 1, turn = ColorType::Black, winner = ColorType::None))]
    pub fn new_for_python(
        board: Board,
        move_number: u16,
        turn: ColorType,
        winner: ColorType,
    ) -> Self {
        Self {
            board,
            move_number,
            turn,
            winner,
        }
    }

    #[pyo3(name = "input_board")]
    pub fn python_input_board(&mut self, sfen: String) {
        self.input_board(sfen);
    }

    #[pyo3(name = "input_hand")]
    pub fn python_input_hand(&mut self, sfen: String) {
        self.input_hand(sfen);
    }

    #[pyo3(name = "input_move_number")]
    pub fn python_input_move_number(&mut self, sfen: String) {
        self.input_move_number(sfen);
    }

    #[pyo3(name = "input_turn")]
    pub fn python_input_turn(&mut self, sfen: String) {
        self.input_turn(sfen);
    }

    #[pyo3(name = "is_finished")]
    pub fn python_is_finished(&self) -> (bool, ColorType) {
        self.is_finished()
    }

    #[pyo3(name = "execute_move")]
    pub fn python_execute_move(&mut self, moves: &Move) {
        self.execute_move(moves);
    }

    #[pyo3(name = "random_play")]
    pub fn python_random_play(&mut self) -> Self {
        self.random_play()
    }

    #[pyo3(name = "generate_random_board")]
    pub fn python_generate_random_board(&mut self) -> Board {
        self.generate_random_board()
    }
}
