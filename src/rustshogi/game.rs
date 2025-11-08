use super::board::Board;
use super::color::{convert_from_string, get_reverse_color, ColorType};
use super::mctsresult::MctsResult;
use super::moves::Move;
use super::piece::Piece;
use super::random::Random;
use num_cpus;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::str::Chars;

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
        let mut current_sfen: Chars = sfen.chars();
        while let Some(ch) = current_sfen.next() {
            if ch.is_ascii_digit() {
                let consecutive: u8 = ch.to_digit(10).unwrap() as u8;
                let piece: Piece = Piece::from_char(current_sfen.next().unwrap());
                self.board
                    .hand
                    .add_pieces(piece.owner, piece.piece_type, consecutive);
            } else {
                let piece: Piece = Piece::from_char(ch);
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
            let moves: Vec<Move> = self.board.search_moves(self.turn, true);
            let amove: &Move = &moves[0];
            self.execute_move(amove);
            let is_finish: (bool, ColorType) = self.is_finished();
            if is_finish.0 {
                self.winner = is_finish.1;
                break;
            }
        }
        self.clone()
    }

    fn perform_random_playout(&mut self, use_cache: bool) -> ColorType {
        while !self.is_finished().0 {
            let moves: Vec<Move> = self.board.search_moves(self.turn, use_cache);
            if moves.is_empty() {
                break;
            }

            let mut random: Random = Random::new(0, (moves.len() - 1) as u16);
            let amove: &Move = &moves[random.generate_one() as usize];
            self.execute_move(amove);
        }
        self.is_finished().1
    }

    pub fn random_play(&mut self) -> Self {
        self.winner = self.perform_random_playout(true);
        self.clone()
    }

    pub fn random_move_parallel(&self, num: usize, num_threads: usize) -> Vec<MctsResult> {
        let pool: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let next_moves: Vec<Move> = self.board.search_moves(self.turn, true);
        let next_move_count: usize = next_moves.len();

        if next_move_count == 0 {
            return vec![];
        }

        // Count how many simulations to run for each move.
        let mut counts: Vec<i32> = vec![0; next_move_count];
        let mut random_gen: Random = Random::new(0, (next_move_count - 1) as u16);
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
                    let mut initial_game_clone: Game = self.clone();
                    initial_game_clone.execute_move(&next_moves[move_index]);

                    // Run simulations for this move sequentially within this parallel task.
                    (0..num_sims_for_this_move)
                        .map(|_| {
                            let mut game_clone = initial_game_clone.clone();
                            let winner = game_clone.perform_random_playout(false);
                            (winner, move_index)
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        });

        // Aggregate results into each MctsResult
        for (winner, move_index) in simulation_results {
            results[move_index].plus_result(winner);
        }

        results
    }

    pub fn generate_random_board(&mut self) -> Board {
        let mut random: Random = Random::new(0, 150);
        let move_count: usize = random.generate_one() as usize;

        for _ in 0..move_count {
            if self.is_finished().0 {
                break;
            }
            let moves: Vec<Move> = self.board.search_moves(self.turn, true);
            if moves.is_empty() {
                break;
            }

            let mut random: Random = Random::new(0, (moves.len() - 1) as u16);
            let amove: &Move = &moves[random.generate_one() as usize].clone();
            self.execute_move(amove);
        }
        self.board.clone()
    }
}

#[pymethods]
impl Game {
    #[pyo3(name = "random_move")]
    #[pyo3(signature = (num, threads = None))]
    pub fn python_random_move(&self, num: usize, threads: Option<usize>) -> Vec<MctsResult> {
        let num_threads: usize = threads.unwrap_or_else(num_cpus::get);
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
