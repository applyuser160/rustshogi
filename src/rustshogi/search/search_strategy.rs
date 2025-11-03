use super::super::board::Board;
use super::super::color::ColorType;
use super::super::evaluator::abst::Evaluator;
use super::super::evaluator::simple::SimpleEvaluator;
use super::super::moves::Move;
use pyo3::prelude::*;

/// 評価結果を表す構造体
#[derive(Debug, Clone)]
#[pyclass]
pub struct EvaluationResult {
    #[pyo3(get, set)]
    pub score: f32,
    #[pyo3(get, set)]
    pub best_move: Option<super::super::moves::Move>,
    #[pyo3(get, set)]
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

/// デフォルト評価関数の取得
pub fn get_default_evaluator() -> SimpleEvaluator {
    SimpleEvaluator::new()
}

/// 手がない場合の処理
pub fn handle_no_moves(
    evaluator: &dyn Evaluator,
    board: &Board,
    color: ColorType,
) -> EvaluationResult {
    EvaluationResult {
        score: evaluator.evaluate(board, color),
        best_move: None,
        nodes_searched: 1,
    }
}

/// 探索の共通初期化処理
/// 評価関数の準備、手の取得、初期化を行い、探索を実行するためのヘルパー関数
pub fn search_helper<F>(
    board: &Board,
    color: ColorType,
    evaluator: Option<&dyn Evaluator>,
    search_fn: F,
) -> EvaluationResult
where
    F: FnOnce(&Board, ColorType, &dyn Evaluator, Vec<Move>) -> EvaluationResult,
{
    let default_evaluator;
    let evaluator_ref = match evaluator {
        Some(e) => e,
        None => {
            default_evaluator = get_default_evaluator();
            &default_evaluator as &dyn Evaluator
        }
    };

    let moves = board.search_moves(color, true);
    if moves.is_empty() {
        return handle_no_moves(evaluator_ref, board, color);
    }

    search_fn(board, color, evaluator_ref, moves)
}
