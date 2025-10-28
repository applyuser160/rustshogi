use super::super::board::Board;
use super::super::color::ColorType;
use super::abst::Evaluator;

/// 簡易評価関数（駒の価値のみを使用）
#[derive(Debug, Clone)]
pub struct SimpleEvaluator {
    pub piece_values: std::collections::HashMap<super::super::piece::PieceType, f32>,
}

impl Default for SimpleEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleEvaluator {
    pub fn new() -> Self {
        let mut piece_values = std::collections::HashMap::new();
        piece_values.insert(super::super::piece::PieceType::King, 10000.0);
        piece_values.insert(super::super::piece::PieceType::Dragon, 12.0);
        piece_values.insert(super::super::piece::PieceType::Horse, 11.0);
        piece_values.insert(super::super::piece::PieceType::Rook, 10.0);
        piece_values.insert(super::super::piece::PieceType::Bichop, 9.0);
        piece_values.insert(super::super::piece::PieceType::Gold, 6.0);
        piece_values.insert(super::super::piece::PieceType::Silver, 5.0);
        piece_values.insert(super::super::piece::PieceType::Knight, 4.0);
        piece_values.insert(super::super::piece::PieceType::Lance, 3.0);
        piece_values.insert(super::super::piece::PieceType::Pawn, 1.0);
        piece_values.insert(super::super::piece::PieceType::ProSilver, 6.0);
        piece_values.insert(super::super::piece::PieceType::ProKnight, 5.0);
        piece_values.insert(super::super::piece::PieceType::ProLance, 4.0);
        piece_values.insert(super::super::piece::PieceType::ProPawn, 3.0);

        Self { piece_values }
    }
}

impl Evaluator for SimpleEvaluator {
    fn evaluate(&self, board: &Board, color: ColorType) -> f32 {
        let mut score = 0.0;

        // 盤上の駒を評価
        for row in 1..=9 {
            for col in 1..=9 {
                let address = super::super::address::Address::from_numbers(col, row);
                let index = address.to_index();
                let piece = board.get_piece(index);

                if piece.piece_type != super::super::piece::PieceType::None {
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

        // 持ち駒を評価
        for color_type in [ColorType::Black, ColorType::White] {
            for piece_type in [
                super::super::piece::PieceType::Rook,
                super::super::piece::PieceType::Bichop,
                super::super::piece::PieceType::Gold,
                super::super::piece::PieceType::Silver,
                super::super::piece::PieceType::Knight,
                super::super::piece::PieceType::Lance,
                super::super::piece::PieceType::Pawn,
            ] {
                let count = board.hand.get_count(color_type, piece_type);
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
