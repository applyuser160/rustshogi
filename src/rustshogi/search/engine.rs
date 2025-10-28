use super::super::board::Board;
use super::super::color::ColorType;
use super::super::evaluator::abst::Evaluator;
use super::super::evaluator::neural::NeuralEvaluator;
use super::super::evaluator::simple::SimpleEvaluator;
use super::alphabeta::AlphaBetaSearchStrategy;
use super::minmax::MinMaxSearchStrategy;
use super::search_strategy::{EvaluationResult, SearchStrategy};
use pyo3::prelude::*;

/// Evaluatorトレイトを持つラッパーenum
/// Pythonから様々な評価器を受け取るための統一的インターフェース
#[derive(FromPyObject)]
pub enum EvaluatorWrapper {
    Simple(SimpleEvaluator),
    Neural(NeuralEvaluator),
}

impl EvaluatorWrapper {
    fn into_evaluator(self) -> Box<dyn Evaluator + Send + Sync> {
        match self {
            EvaluatorWrapper::Simple(e) => Box::new(e),
            EvaluatorWrapper::Neural(e) => Box::new(e),
        }
    }
}

/// 探索エンジン
/// 探索戦略と評価関数を組み合わせて使用
#[pyclass]
pub struct SearchEngine {
    search_strategy: Box<dyn SearchStrategy + Send + Sync>,
    evaluator: Option<Box<dyn Evaluator + Send + Sync>>,
}

impl SearchEngine {
    pub fn new(
        search_strategy: Box<dyn SearchStrategy + Send + Sync>,
        evaluator: Option<Box<dyn Evaluator + Send + Sync>>,
    ) -> Self {
        Self {
            search_strategy,
            evaluator,
        }
    }

    /// 探索を実行する
    pub fn search(&self, board: &Board, color: ColorType, depth: u8) -> EvaluationResult {
        self.search_strategy.search(
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
    #[pyo3(signature = (algorithm="minmax".to_string(), max_nodes=1000000, evaluator=None))]
    pub fn new_for_python(
        algorithm: String,
        max_nodes: u64,
        evaluator: Option<EvaluatorWrapper>,
    ) -> PyResult<Self> {
        let search_strategy: Box<dyn SearchStrategy + Send + Sync> =
            match algorithm.to_lowercase().as_str() {
                "alphabeta" => Box::new(AlphaBetaSearchStrategy::new(max_nodes)),
                _ => Box::new(MinMaxSearchStrategy::new(max_nodes)),
            };

        // evaluatorがNoneの場合はデフォルトでSimpleEvaluatorを使用
        let evaluator: Option<Box<dyn Evaluator + Send + Sync>> = evaluator
            .map(|e| e.into_evaluator())
            .or_else(|| Some(Box::new(SimpleEvaluator::new())));

        Ok(Self::new(search_strategy, evaluator))
    }

    #[pyo3(name = "search")]
    pub fn python_search(&self, board: &Board, color: ColorType, depth: u8) -> EvaluationResult {
        self.search(board, color, depth)
    }
}
