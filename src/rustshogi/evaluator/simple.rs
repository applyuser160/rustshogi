use super::super::address::Address;
use super::super::board::Board;
use super::super::color::ColorType;
use super::super::piece::{Piece, PieceType};
use super::abst::Evaluator;
use pyo3::prelude::*;

use std::collections::HashMap;
/// Simple evaluation function (uses only piece values)
#[pyclass]
#[derive(Debug, Clone)]
pub struct SimpleEvaluator {
    pub piece_values: HashMap<PieceType, f32>,
}

impl Default for SimpleEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl SimpleEvaluator {
    #[new]
    pub fn new() -> Self {
        let mut piece_values: HashMap<PieceType, f32> = HashMap::new();
        piece_values.insert(PieceType::King, 10000.0);
        piece_values.insert(PieceType::Dragon, 12.0);
        piece_values.insert(PieceType::Horse, 11.0);
        piece_values.insert(PieceType::Rook, 10.0);
        piece_values.insert(PieceType::Bishop, 9.0);
        piece_values.insert(PieceType::Gold, 6.0);
        piece_values.insert(PieceType::Silver, 5.0);
        piece_values.insert(PieceType::Knight, 4.0);
        piece_values.insert(PieceType::Lance, 3.0);
        piece_values.insert(PieceType::Pawn, 1.0);
        piece_values.insert(PieceType::ProSilver, 6.0);
        piece_values.insert(PieceType::ProKnight, 5.0);
        piece_values.insert(PieceType::ProLance, 4.0);
        piece_values.insert(PieceType::ProPawn, 3.0);

        Self { piece_values }
    }

    #[pyo3(name = "evaluate")]
    pub fn python_evaluate(&self, board: &Board, color: ColorType) -> f32 {
        self.evaluate(board, color)
    }
}

impl Evaluator for SimpleEvaluator {
    fn evaluate(&self, board: &Board, color: ColorType) -> f32 {
        let mut score = 0.0;

        // Evaluate pieces on the board
        for row in 1..=9 {
            for col in 1..=9 {
                let address: Address = Address::from_numbers(col, row);
                let index: u8 = address.to_index();
                let piece: Piece = board.get_piece(index);

                if piece.piece_type != PieceType::None {
                    if let Some(&value) = self.piece_values.get(&piece.piece_type) {
                        if piece.owner == color {
                            score += value;
                        } else {
                            score -= value;
                        }
                    }
                }
            }
        }

        // Evaluate pieces in hand
        for color_type in [ColorType::Black, ColorType::White] {
            for piece_type in [
                super::super::piece::PieceType::Rook,
                PieceType::Bishop,
                PieceType::Gold,
                PieceType::Silver,
                PieceType::Knight,
                PieceType::Lance,
                PieceType::Pawn,
            ] {
                let count: u8 = board.hand.get_count(color_type, piece_type);
                if count > 0 {
                    if let Some(&value) = self.piece_values.get(&piece_type) {
                        if color_type == color {
                            score += value * count as f32;
                        } else {
                            score -= value * count as f32;
                        }
                    }
                }
            }
        }

        score
    }
}
