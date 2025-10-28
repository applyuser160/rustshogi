use super::super::board::Board;
use super::super::color::ColorType;
use super::super::evaluator::Evaluator;
use super::super::moves::Move;
use super::super::search_strategy::{EvaluationResult, SearchStrategy};
use super::alphabeta::AlphaBetaSearch;
use super::minmax::MinMaxSearch;
use pyo3::prelude::*;

/// 探索エンジン
/// 探索戦略と評価関数を組み合わせて使用
#[pyclass]
pub struct SearchEngine {
    strategy: Box<dyn SearchStrategy + Send + Sync>,
    evaluator: Option<Box<dyn Evaluator + Send + Sync>>,
}

impl SearchEngine {
    pub fn new(
        strategy: Box<dyn SearchStrategy + Send + Sync>,
        evaluator: Option<Box<dyn Evaluator + Send + Sync>>,
    ) -> Self {
        Self {
            strategy,
            evaluator,
        }
    }

    /// 探索を実行する
    pub fn search(&self, board: &Board, color: ColorType, depth: u8) -> EvaluationResult {
        self.strategy.search(
            board,
            color,
            depth,
            self.evaluator
                .as_ref()
                .map(|e| e.as_ref() as &dyn Evaluator),
        )
    }
}

#[pymethods]
impl SearchEngine {
    #[new]
    #[pyo3(signature = (algorithm="minmax".to_string(), max_nodes=1000000))]
    pub fn new_for_python(algorithm: String, max_nodes: u64) -> Self {
        let strategy: Box<dyn SearchStrategy + Send + Sync> =
            match algorithm.to_lowercase().as_str() {
                "alphabeta" => Box::new(AlphaBetaSearch::new(max_nodes)),
                _ => Box::new(MinMaxSearch::new(max_nodes)),
            };

        let evaluator: Option<Box<dyn Evaluator + Send + Sync>> = Some(Box::new(
            super::super::evaluator::simple::SimpleEvaluator::new(),
        ));

        Self::new(strategy, evaluator)
    }

    #[pyo3(name = "search")]
    pub fn python_search(&self, board: &Board, color: ColorType, depth: u8) -> (f32, Option<Move>) {
        let result = self.search(board, color, depth);
        (result.score, result.best_move)
    }

    #[pyo3(name = "search_full")]
    pub fn python_search_full(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
    ) -> (f32, Option<Move>, u64) {
        let result = self.search(board, color, depth);
        (result.score, result.best_move, result.nodes_searched)
    }
}
