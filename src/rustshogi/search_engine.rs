//! 将棋AIの探索エンジン設計
//! 
//! アーキテクチャ概要:
//! ```
//! SearchEngine
//! ├── SearchAlgorithm (トレイト)
//! │   ├── Minimax
//! │   ├── AlphaBeta
//! │   └── MonteCarlo
//! └── Evaluator (トレイト)
//!     ├── SimpleEvaluator
//!     ├── MaterialEvaluator
//!     └── NeuralNetworkEvaluator
//! ```
//! 
//! 設計思想:
//! - 探索アルゴリズムと評価関数を分離し、組み合わせ可能にする
//! - トレイトを使用してプラグイン的な拡張を可能にする
//! - 将来的に外部クレートとして追加できる設計

use std::fmt;

/// 将棋の手を表現する構造体（スタブ）
#[derive(Debug, Clone, PartialEq)]
pub struct Move {
    pub from: Option<(u8, u8)>,  // 移動元（打ち駒の場合はNone）
    pub to: (u8, u8),            // 移動先
    pub piece: PieceType,        // 駒の種類
    pub promotion: bool,         // 成りかどうか
}

/// 駒の種類（スタブ）
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PieceType {
    Pawn,    // 歩
    Lance,   // 香
    Knight,  // 桂
    Silver,  // 銀
    Gold,    // 金
    Bishop,  // 角
    Rook,    // 飛
    King,    // 王
}

/// 手番
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
    Black,  // 先手
    White,  // 後手
}

impl Color {
    /// 相手の手番を取得
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

/// 将棋の局面を表現する構造体（スタブ）
/// 実際の実装では、より詳細な盤面情報が必要
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    /// 盤面の駒配置（9x9の配列）
    /// None: 空きマス, Some((駒の種類, 手番))
    pub board: [[Option<(PieceType, Color)>; 9]; 9],
    /// 現在の手番
    pub current_player: Color,
    /// 先手の持ち駒
    pub black_hand: Vec<PieceType>,
    /// 後手の持ち駒
    pub white_hand: Vec<PieceType>,
}

impl Position {
    /// 新しい初期局面を作成
    pub fn new() -> Self {
        // 簡略化された初期局面（実際はもっと複雑）
        let mut board = [[None; 9]; 9];
        
        // 王を配置（簡略化）
        board[0][4] = Some((PieceType::King, Color::Black));
        board[8][4] = Some((PieceType::King, Color::White));
        
        Self {
            board,
            current_player: Color::Black,
            black_hand: Vec::new(),
            white_hand: Vec::new(),
        }
    }
    
    /// 手を適用して新しい局面を作成
    pub fn make_move(&self, mv: &Move) -> Self {
        let mut new_pos = self.clone();
        
        // 手を適用（簡略化された実装）
        if let Some((from_x, from_y)) = mv.from {
            new_pos.board[from_y as usize][from_x as usize] = None;
        }
        new_pos.board[mv.to.1 as usize][mv.to.0 as usize] = Some((mv.piece, self.current_player));
        
        // 手番を変更
        new_pos.current_player = self.current_player.opponent();
        
        new_pos
    }
    
    /// 合法手を生成（簡略化された実装）
    pub fn generate_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        
        // 簡略化: 空いているマスに王を移動する手を生成
        for y in 0..9 {
            for x in 0..9 {
                if self.board[y][x].is_none() {
                    moves.push(Move {
                        from: Some((4, 4)), // 王の位置から
                        to: (x as u8, y as u8),
                        piece: PieceType::King,
                        promotion: false,
                    });
                }
            }
        }
        
        moves
    }
    
    /// ゲーム終了判定（簡略化）
    pub fn is_game_over(&self) -> bool {
        // 簡略化: 王がいない場合は終了
        let mut has_black_king = false;
        let mut has_white_king = false;
        
        for row in &self.board {
            for cell in row {
                if let Some((PieceType::King, color)) = cell {
                    match color {
                        Color::Black => has_black_king = true,
                        Color::White => has_white_king = true,
                    }
                }
            }
        }
        
        !has_black_king || !has_white_king
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

/// 探索結果
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 最善手
    pub best_move: Option<Move>,
    /// 評価値
    pub score: f64,
    /// 探索したノード数
    pub nodes_searched: u64,
    /// 探索時間（ミリ秒）
    pub search_time_ms: u64,
}

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SearchResult {{ best_move: {:?}, score: {:.2}, nodes: {}, time: {}ms }}",
               self.best_move, self.score, self.nodes_searched, self.search_time_ms)
    }
}

/// 局面評価を行うトレイト
/// 異なる評価関数をプラグイン的に切り替え可能
pub trait Evaluator: Send + Sync {
    /// 局面を評価する
    /// 正の値: 先手有利、負の値: 後手有利
    fn evaluate(&self, position: &Position) -> f64;
    
    /// 評価関数の名前を取得
    fn name(&self) -> &str;
}

/// 探索アルゴリズムを定義するトレイト
/// 異なる探索手法をプラグイン的に切り替え可能
pub trait SearchAlgorithm: Send + Sync {
    /// 指定された深度で探索を実行
    /// 
    /// # Arguments
    /// * `position` - 探索開始局面
    /// * `depth` - 探索深度
    /// * `evaluator` - 使用する評価関数
    /// 
    /// # Returns
    /// 探索結果
    fn search(&self, position: &Position, depth: u8, evaluator: &dyn Evaluator) -> SearchResult;
    
    /// 探索アルゴリズムの名前を取得
    fn name(&self) -> &str;
}

/// 探索エンジンのメイン構造体
/// SearchAlgorithmとEvaluatorを組み合わせて動作
pub struct SearchEngine {
    algorithm: Box<dyn SearchAlgorithm>,
    evaluator: Box<dyn Evaluator>,
}

impl SearchEngine {
    /// 新しい探索エンジンを作成
    /// 
    /// # Arguments
    /// * `algorithm` - 使用する探索アルゴリズム
    /// * `evaluator` - 使用する評価関数
    pub fn new(algorithm: Box<dyn SearchAlgorithm>, evaluator: Box<dyn Evaluator>) -> Self {
        Self {
            algorithm,
            evaluator,
        }
    }
    
    /// 探索を実行
    /// 
    /// # Arguments
    /// * `position` - 探索開始局面
    /// * `depth` - 探索深度
    /// 
    /// # Returns
    /// 探索結果
    pub fn search(&self, position: &Position, depth: u8) -> SearchResult {
        self.algorithm.search(position, depth, self.evaluator.as_ref())
    }
    
    /// 探索アルゴリズムを取得
    pub fn algorithm_name(&self) -> &str {
        self.algorithm.name()
    }
    
    /// 評価関数を取得
    pub fn evaluator_name(&self) -> &str {
        self.evaluator.name()
    }
    
    /// 探索アルゴリズムを変更
    pub fn set_algorithm(&mut self, algorithm: Box<dyn SearchAlgorithm>) {
        self.algorithm = algorithm;
    }
    
    /// 評価関数を変更
    pub fn set_evaluator(&mut self, evaluator: Box<dyn Evaluator>) {
        self.evaluator = evaluator;
    }
}

impl fmt::Display for SearchEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SearchEngine {{ algorithm: {}, evaluator: {} }}",
               self.algorithm_name(), self.evaluator_name())
    }
}

// ============================================================================
// 探索アルゴリズムの実装
// ============================================================================

/// Minimax探索アルゴリズム
pub struct MinimaxSearch;

impl SearchAlgorithm for MinimaxSearch {
    fn search(&self, position: &Position, depth: u8, evaluator: &dyn Evaluator) -> SearchResult {
        let start_time = std::time::Instant::now();
        let mut nodes_searched = 0u64;
        
        let (best_move, score) = self.minimax(position, depth, true, evaluator, &mut nodes_searched);
        
        SearchResult {
            best_move,
            score,
            nodes_searched,
            search_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    fn name(&self) -> &str {
        "Minimax"
    }
}

impl MinimaxSearch {
    /// Minimax探索の再帰実装
    fn minimax(
        &self,
        position: &Position,
        depth: u8,
        is_maximizing: bool,
        evaluator: &dyn Evaluator,
        nodes_searched: &mut u64,
    ) -> (Option<Move>, f64) {
        *nodes_searched += 1;
        
        // 終了条件
        if depth == 0 || position.is_game_over() {
            return (None, evaluator.evaluate(position));
        }
        
        let moves = position.generate_moves();
        if moves.is_empty() {
            return (None, evaluator.evaluate(position));
        }
        
        let mut best_move = None;
        let mut best_score = if is_maximizing { f64::NEG_INFINITY } else { f64::INFINITY };
        
        for mv in moves {
            let new_position = position.make_move(&mv);
            let (_, score) = self.minimax(&new_position, depth - 1, !is_maximizing, evaluator, nodes_searched);
            
            if is_maximizing {
                if score > best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
            } else {
                if score < best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
            }
        }
        
        (best_move, best_score)
    }
}

/// Alpha-Beta探索アルゴリズム
pub struct AlphaBetaSearch;

impl SearchAlgorithm for AlphaBetaSearch {
    fn search(&self, position: &Position, depth: u8, evaluator: &dyn Evaluator) -> SearchResult {
        let start_time = std::time::Instant::now();
        let mut nodes_searched = 0u64;
        
        let (best_move, score) = self.alpha_beta(
            position, 
            depth, 
            f64::NEG_INFINITY, 
            f64::INFINITY, 
            true, 
            evaluator, 
            &mut nodes_searched
        );
        
        SearchResult {
            best_move,
            score,
            nodes_searched,
            search_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    fn name(&self) -> &str {
        "Alpha-Beta"
    }
}

impl AlphaBetaSearch {
    /// Alpha-Beta探索の再帰実装
    fn alpha_beta(
        &self,
        position: &Position,
        depth: u8,
        mut alpha: f64,
        mut beta: f64,
        is_maximizing: bool,
        evaluator: &dyn Evaluator,
        nodes_searched: &mut u64,
    ) -> (Option<Move>, f64) {
        *nodes_searched += 1;
        
        // 終了条件
        if depth == 0 || position.is_game_over() {
            return (None, evaluator.evaluate(position));
        }
        
        let moves = position.generate_moves();
        if moves.is_empty() {
            return (None, evaluator.evaluate(position));
        }
        
        let mut best_move = None;
        let mut best_score = if is_maximizing { f64::NEG_INFINITY } else { f64::INFINITY };
        
        for mv in moves {
            let new_position = position.make_move(&mv);
            let (_, score) = self.alpha_beta(
                &new_position, 
                depth - 1, 
                alpha, 
                beta, 
                !is_maximizing, 
                evaluator, 
                nodes_searched
            );
            
            if is_maximizing {
                if score > best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
                alpha = alpha.max(score);
                if beta <= alpha {
                    break; // Beta cut-off
                }
            } else {
                if score < best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
                beta = beta.min(score);
                if beta <= alpha {
                    break; // Alpha cut-off
                }
            }
        }
        
        (best_move, best_score)
    }
}

// ============================================================================
// 評価関数の実装
// ============================================================================

/// シンプルな評価関数
/// 駒の価値のみを考慮した基本的な評価
pub struct SimpleEvaluator;

impl Evaluator for SimpleEvaluator {
    fn evaluate(&self, position: &Position) -> f64 {
        let mut score = 0.0;
        
        // 駒の価値を計算
        for row in &position.board {
            for cell in row {
                if let Some((piece_type, color)) = cell {
                    let piece_value = self.get_piece_value(*piece_type);
                    let value = match color {
                        Color::Black => piece_value,
                        Color::White => -piece_value,
                    };
                    score += value;
                }
            }
        }
        
        score
    }
    
    fn name(&self) -> &str {
        "Simple"
    }
}

impl SimpleEvaluator {
    /// 駒の価値を取得
    fn get_piece_value(&self, piece_type: PieceType) -> f64 {
        match piece_type {
            PieceType::Pawn => 1.0,
            PieceType::Lance => 3.0,
            PieceType::Knight => 3.0,
            PieceType::Silver => 5.0,
            PieceType::Gold => 6.0,
            PieceType::Bishop => 8.0,
            PieceType::Rook => 10.0,
            PieceType::King => 1000.0,
        }
    }
}

/// マテリアル評価関数
/// 駒の価値と位置を考慮した評価
pub struct MaterialEvaluator;

impl Evaluator for MaterialEvaluator {
    fn evaluate(&self, position: &Position) -> f64 {
        let mut score = 0.0;
        
        // 駒の価値を計算
        for (y, row) in position.board.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                if let Some((piece_type, color)) = cell {
                    let piece_value = self.get_piece_value(*piece_type);
                    let position_bonus = self.get_position_bonus(*piece_type, x, y, *color);
                    let total_value = piece_value + position_bonus;
                    
                    let value = match color {
                        Color::Black => total_value,
                        Color::White => -total_value,
                    };
                    score += value;
                }
            }
        }
        
        score
    }
    
    fn name(&self) -> &str {
        "Material"
    }
}

impl MaterialEvaluator {
    /// 駒の基本価値を取得
    fn get_piece_value(&self, piece_type: PieceType) -> f64 {
        match piece_type {
            PieceType::Pawn => 1.0,
            PieceType::Lance => 3.0,
            PieceType::Knight => 3.0,
            PieceType::Silver => 5.0,
            PieceType::Gold => 6.0,
            PieceType::Bishop => 8.0,
            PieceType::Rook => 10.0,
            PieceType::King => 1000.0,
        }
    }
    
    /// 位置ボーナスを計算
    fn get_position_bonus(&self, piece_type: PieceType, x: usize, y: usize, _color: Color) -> f64 {
        // 簡略化: 中央に近いほど価値が高い
        let center_x = 4.0;
        let center_y = 4.0;
        let distance = ((x as f64 - center_x).powi(2) + (y as f64 - center_y).powi(2)).sqrt();
        let max_distance = ((4.0_f64).powi(2) + (4.0_f64).powi(2)).sqrt();
        
        let position_factor = 1.0 - (distance / max_distance);
        
        // 駒の種類に応じて位置の重要度を調整
        let importance = match piece_type {
            PieceType::King => 0.5,  // 王は位置よりも安全が重要
            PieceType::Pawn => 0.3,  // 歩は前進が重要
            _ => 0.2,                // その他の駒
        };
        
        position_factor * importance
    }
}

/// Monte Carlo探索アルゴリズム
/// ランダムな手順でゲームを終了までシミュレートし、勝率を基に評価
pub struct MonteCarloSearch {
    simulations: u32,  // シミュレーション回数
}

impl MonteCarloSearch {
    pub fn new(simulations: u32) -> Self {
        Self { simulations }
    }
}

impl SearchAlgorithm for MonteCarloSearch {
    fn search(&self, position: &Position, _depth: u8, evaluator: &dyn Evaluator) -> SearchResult {
        let start_time = std::time::Instant::now();
        let mut nodes_searched = 0u64;
        
        let moves = position.generate_moves();
        if moves.is_empty() {
            return SearchResult {
                best_move: None,
                score: evaluator.evaluate(position),
                nodes_searched: 1,
                search_time_ms: start_time.elapsed().as_millis() as u64,
            };
        }
        
        let mut best_move = None;
        let mut best_score = f64::NEG_INFINITY;
        
        // 各手についてMonte Carloシミュレーションを実行
        for mv in moves {
            let mut total_score = 0.0;
            
            for _ in 0..self.simulations {
                let new_position = position.make_move(&mv);
                let score = self.simulate_random_game(&new_position, evaluator, &mut nodes_searched);
                total_score += score;
            }
            
            let average_score = total_score / self.simulations as f64;
            
            if average_score > best_score {
                best_score = average_score;
                best_move = Some(mv);
            }
        }
        
        SearchResult {
            best_move,
            score: best_score,
            nodes_searched,
            search_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }
    
    fn name(&self) -> &str {
        "Monte Carlo"
    }
}

impl MonteCarloSearch {
    /// ランダムな手順でゲームを終了までシミュレート
    fn simulate_random_game(
        &self,
        position: &Position,
        evaluator: &dyn Evaluator,
        nodes_searched: &mut u64,
    ) -> f64 {
        *nodes_searched += 1;
        
        if position.is_game_over() {
            return evaluator.evaluate(position);
        }
        
        let moves = position.generate_moves();
        if moves.is_empty() {
            return evaluator.evaluate(position);
        }
        
        // ランダムに手を選択
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_index = rng.gen_range(0..moves.len());
        let random_move = &moves[random_index];
        
        let new_position = position.make_move(random_move);
        self.simulate_random_game(&new_position, evaluator, nodes_searched)
    }
}

// ============================================================================
// 使用例とテスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let position = Position::new();
        assert_eq!(position.current_player, Color::Black);
        assert!(position.board[0][4] == Some((PieceType::King, Color::Black)));
        assert!(position.board[8][4] == Some((PieceType::King, Color::White)));
    }

    #[test]
    fn test_move_generation() {
        let position = Position::new();
        let moves = position.generate_moves();
        assert!(!moves.is_empty());
    }

    #[test]
    fn test_simple_evaluator() {
        let position = Position::new();
        let evaluator = SimpleEvaluator;
        let score = evaluator.evaluate(&position);
        // 初期局面では先手と後手の駒の価値が同じなので0に近いはず
        assert!(score.abs() < 1.0);
    }

    #[test]
    fn test_minimax_search() {
        let position = Position::new();
        let engine = SearchEngine::new(
            Box::new(MinimaxSearch),
            Box::new(SimpleEvaluator),
        );
        
        let result = engine.search(&position, 2);
        println!("Minimax search result: {}", result);
        assert!(result.nodes_searched > 0);
    }

    #[test]
    fn test_alpha_beta_search() {
        let position = Position::new();
        let engine = SearchEngine::new(
            Box::new(AlphaBetaSearch),
            Box::new(SimpleEvaluator),
        );
        
        let result = engine.search(&position, 2);
        println!("Alpha-Beta search result: {}", result);
        assert!(result.nodes_searched > 0);
    }

    #[test]
    fn test_algorithm_comparison() {
        let position = Position::new();
        
        // Minimax探索
        let minimax_engine = SearchEngine::new(
            Box::new(MinimaxSearch),
            Box::new(SimpleEvaluator),
        );
        let minimax_result = minimax_engine.search(&position, 2);
        
        // Alpha-Beta探索
        let alphabeta_engine = SearchEngine::new(
            Box::new(AlphaBetaSearch),
            Box::new(SimpleEvaluator),
        );
        let alphabeta_result = alphabeta_engine.search(&position, 2);
        
        println!("Minimax: {}", minimax_result);
        println!("Alpha-Beta: {}", alphabeta_result);
        
        // Alpha-Betaは枝刈りにより探索ノード数が少ないはず
        assert!(alphabeta_result.nodes_searched <= minimax_result.nodes_searched);
    }

    #[test]
    fn test_evaluator_comparison() {
        let position = Position::new();
        
        // Simple評価関数
        let simple_engine = SearchEngine::new(
            Box::new(MinimaxSearch),
            Box::new(SimpleEvaluator),
        );
        let simple_result = simple_engine.search(&position, 2);
        
        // Material評価関数
        let material_engine = SearchEngine::new(
            Box::new(MinimaxSearch),
            Box::new(MaterialEvaluator),
        );
        let material_result = material_engine.search(&position, 2);
        
        println!("Simple evaluator: {}", simple_result);
        println!("Material evaluator: {}", material_result);
        
        // 両方とも有効な結果を返すはず
        assert!(simple_result.nodes_searched > 0);
        assert!(material_result.nodes_searched > 0);
    }

    #[test]
    fn test_engine_switching() {
        let mut engine = SearchEngine::new(
            Box::new(MinimaxSearch),
            Box::new(SimpleEvaluator),
        );
        
        assert_eq!(engine.algorithm_name(), "Minimax");
        assert_eq!(engine.evaluator_name(), "Simple");
        
        // アルゴリズムを変更
        engine.set_algorithm(Box::new(AlphaBetaSearch));
        assert_eq!(engine.algorithm_name(), "Alpha-Beta");
        
        // 評価関数を変更
        engine.set_evaluator(Box::new(MaterialEvaluator));
        assert_eq!(engine.evaluator_name(), "Material");
    }

    #[test]
    fn test_monte_carlo_search() {
        let position = Position::new();
        let engine = SearchEngine::new(
            Box::new(MonteCarloSearch::new(100)), // 100回シミュレーション
            Box::new(SimpleEvaluator),
        );
        
        let result = engine.search(&position, 2);
        println!("Monte Carlo search result: {}", result);
        assert!(result.nodes_searched > 0);
    }

    #[test]
    fn test_all_algorithms_comparison() {
        let position = Position::new();
        
        // 各アルゴリズムで探索
        let algorithms: Vec<(&str, Box<dyn SearchAlgorithm>)> = vec![
            ("Minimax", Box::new(MinimaxSearch)),
            ("Alpha-Beta", Box::new(AlphaBetaSearch)),
            ("Monte Carlo", Box::new(MonteCarloSearch::new(50))),
        ];
        
        for (name, algorithm) in algorithms {
            let engine = SearchEngine::new(algorithm, Box::new(SimpleEvaluator));
            let result = engine.search(&position, 2);
            println!("{}: {} (ノード数: {})", name, result.score, result.nodes_searched);
        }
    }
}

/// 使用例を示すサンプル関数
pub fn run_search_example() {
    println!("=== 将棋AI探索エンジン使用例 ===");
    
    // 初期局面を作成
    let position = Position::new();
    println!("初期局面を作成しました");
    
    // Minimax + Simple評価関数の組み合わせ
    let mut engine = SearchEngine::new(
        Box::new(MinimaxSearch),
        Box::new(SimpleEvaluator),
    );
    
    println!("\n1. Minimax + Simple評価関数:");
    println!("エンジン: {}", engine);
    
    let result = engine.search(&position, 2);
    println!("探索結果: {}", result);
    
    // Alpha-Beta + Material評価関数の組み合わせ
    engine.set_algorithm(Box::new(AlphaBetaSearch));
    engine.set_evaluator(Box::new(MaterialEvaluator));
    
    println!("\n2. Alpha-Beta + Material評価関数:");
    println!("エンジン: {}", engine);
    
    let result = engine.search(&position, 2);
    println!("探索結果: {}", result);
    
    // 異なる深度での探索
    println!("\n3. 異なる深度での探索比較:");
    for depth in 1..=3 {
        let result = engine.search(&position, depth);
        println!("深度 {}: {} (ノード数: {})", depth, result.score, result.nodes_searched);
    }
    
    // Monte Carlo探索の例
    engine.set_algorithm(Box::new(MonteCarloSearch::new(100)));
    engine.set_evaluator(Box::new(SimpleEvaluator));
    
    println!("\n4. Monte Carlo探索:");
    println!("エンジン: {}", engine);
    let result = engine.search(&position, 2);
    println!("探索結果: {}", result);
    
    // 全アルゴリズムの比較
    println!("\n5. 全アルゴリズムの比較:");
    let algorithms: Vec<(&str, Box<dyn SearchAlgorithm>)> = vec![
        ("Minimax", Box::new(MinimaxSearch)),
        ("Alpha-Beta", Box::new(AlphaBetaSearch)),
        ("Monte Carlo", Box::new(MonteCarloSearch::new(50))),
    ];
    
    for (name, algorithm) in algorithms {
        let engine = SearchEngine::new(algorithm, Box::new(SimpleEvaluator));
        let result = engine.search(&position, 2);
        println!("{}: {} (ノード数: {}, 時間: {}ms)", 
                name, result.score, result.nodes_searched, result.search_time_ms);
    }
    
    println!("\n=== 探索エンジンの設計完了 ===");
    println!("- SearchAlgorithmトレイト: 探索手法を抽象化");
    println!("- Evaluatorトレイト: 評価関数を抽象化");
    println!("- SearchEngine: アルゴリズムと評価関数を組み合わせ");
    println!("- プラグイン的な拡張が可能な設計");
    println!("- 実装済みアルゴリズム: Minimax, Alpha-Beta, Monte Carlo");
    println!("- 実装済み評価関数: Simple, Material");
}