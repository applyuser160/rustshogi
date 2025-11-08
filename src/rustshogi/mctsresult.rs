use super::board::Board;
use super::color::ColorType;
use super::moves::Move;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct MctsResult {
    #[pyo3(get, set)]
    pub board: Board,
    #[pyo3(get, set)]
    pub mv: Move,
    #[pyo3(get, set)]
    pub white_wins: u64,
    #[pyo3(get, set)]
    pub black_wins: u64,
    #[pyo3(get, set)]
    pub total_games: u64,
}

#[pymethods]
impl MctsResult {
    pub fn merge(&mut self, other: &MctsResult) {
        self.white_wins += other.white_wins;
        self.black_wins += other.black_wins;
        self.total_games += other.total_games;
    }
}

impl Default for MctsResult {
    fn default() -> Self {
        Self::new()
    }
}

impl MctsResult {
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            mv: Move::new(),
            white_wins: 0,
            black_wins: 0,
            total_games: 0,
        }
    }

    pub fn from(board: Board, mv: Move) -> Self {
        Self {
            board,
            mv,
            white_wins: 0,
            black_wins: 0,
            total_games: 0,
        }
    }

    pub fn plus_result(&mut self, winner: ColorType) {
        match winner {
            ColorType::White => self.white_wins += 1,
            ColorType::Black => self.black_wins += 1,
            _ => {}
        }
        self.total_games += 1;
    }

    pub fn calc_result(&self, turn: ColorType) -> f64 {
        if self.total_games == 0 {
            return 0.0;
        }

        let current_turn_wins: f64 = if turn == ColorType::White {
            self.white_wins as f64
        } else {
            self.black_wins as f64
        };

        current_turn_wins / self.total_games as f64
    }

    pub fn print_result(&self) {
        println!("{} times played!", self.total_games);
        println!("bestmove {}", self.mv.to_string());
        println!(
            "White wins: {}, Black wins: {}",
            self.white_wins, self.black_wins
        );
    }
}
