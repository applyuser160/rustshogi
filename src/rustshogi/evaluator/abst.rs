use super::super::board::Board;
use super::super::color::ColorType;

/// 評価関数のトレイト
/// 各評価関数はこのトレイトを実装することで、探索エンジンで使用可能
pub trait Evaluator: Send + Sync {
    /// 盤面を評価する
    ///
    /// # Arguments
    /// * `board` - 評価する盤面
    /// * `color` - 評価するプレイヤーの色
    ///
    /// # Returns
    /// 評価値（colorの視点での評価、大きい方が有利）
    fn evaluate(&self, board: &Board, color: ColorType) -> f32;
}
