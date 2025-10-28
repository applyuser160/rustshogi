use super::super::board::Board;
use super::super::color::ColorType;
use super::super::evaluator::abst::Evaluator;

/// 評価結果を表す構造体
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub score: f32,
    pub best_move: Option<super::super::moves::Move>,
    pub nodes_searched: u64,
}

/// 探索戦略のトレイト
pub trait SearchStrategy {
    /// 探索を実行する
    ///
    /// # Arguments
    /// * `board` - 現在の盤面
    /// * `color` - 手番の色
    /// * `depth` - 探索深度
    /// * `evaluator` - 評価関数（オプション）
    ///
    /// # Returns
    /// 評価結果
    fn search(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        evaluator: Option<&dyn Evaluator>,
    ) -> EvaluationResult;
}
