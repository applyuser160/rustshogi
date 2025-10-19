use super::board::Board;
use super::color::ColorType;
use super::game::Game;
use super::nn_model::{NnModel, NnModelConfig, TrainingConfig, TrainingData};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use pyo3::prelude::*;
use rand;
use rusqlite::{Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 学習データベースのレコード構造体
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingRecord {
    pub id: i64,
    pub board: String, // SFEN形式の文字列
    pub white_wins: i32,
    pub black_wins: i32,
    pub total_games: i32,
    pub created_at: String,
    pub updated_at: String,
}

/// 評価関数システム
#[pyclass]
pub struct Evaluator {
    db_path: String,
}

impl Evaluator {
    /// 新しい評価関数システムを作成
    pub fn new(db_path: String) -> Self {
        Self { db_path }
    }

    /// データベース接続を取得
    fn get_connection(&self) -> SqlResult<Connection> {
        Connection::open(&self.db_path)
    }

    /// データベーステーブルを初期化
    pub fn init_database(&self) -> SqlResult<()> {
        let conn = self.get_connection()?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board TEXT NOT NULL,
                white_wins INTEGER DEFAULT 0,
                black_wins INTEGER DEFAULT 0,
                total_games INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_data_total_games ON training_data(total_games)",
            [],
        )?;

        println!("データベーステーブルを初期化しました: {}", self.db_path);
        Ok(())
    }

    /// ランダム盤面を生成してRDBに保存
    ///
    /// # Arguments
    /// * `count` - 生成する盤面の数
    ///
    /// # Returns
    /// * `SqlResult<i32>` - 保存されたレコード数
    pub fn generate_and_save_random_boards(&self, count: usize) -> SqlResult<i32> {
        let conn = self.get_connection()?;
        let mut saved_count = 0;

        println!("{}個のランダム盤面を生成中...", count);

        for i in 0..count {
            let mut game = Game::new();
            let random_board = game.generate_random_board();
            let board_sfen = random_board.to_string();

            conn.execute(
                "INSERT INTO training_data (board) VALUES (?1)",
                [&board_sfen],
            )?;

            saved_count += 1;

            if (i + 1) % 100 == 0 {
                println!("{}個の盤面を生成・保存しました", i + 1);
            }
        }

        println!("{}個のランダム盤面を生成・保存しました", saved_count);
        Ok(saved_count)
    }

    /// 保存されたレコードを読み取り、ランダム対局を実行して勝利数を更新
    ///
    /// # Arguments
    /// * `trials_per_record` - 各レコードに対する試行回数
    /// * `max_records` - 処理する最大レコード数（Noneの場合は全て）
    ///
    /// # Returns
    /// * `SqlResult<i32>` - 更新されたレコード数
    pub fn update_records_with_random_games(
        &self,
        trials_per_record: usize,
        max_records: Option<usize>,
    ) -> SqlResult<i32> {
        let conn = self.get_connection()?;

        let query = "SELECT id, board FROM training_data ORDER BY total_games ASC, id ASC";
        let mut stmt = conn.prepare(query)?;

        let mut records: Vec<(i64, String)> = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        for row_result in rows {
            let (id, board_sfen) = row_result?;
            records.push((id, board_sfen));
            if let Some(max) = max_records {
                if records.len() >= max {
                    break;
                }
            }
        }

        let mut updated_count = 0;

        for (id, board_sfen) in records {
            // SFEN形式からBoardを復元
            let board = Board::from_sfen(board_sfen);
            let (white_wins, black_wins) = self.run_random_games(&board, trials_per_record);

            conn.execute(
                "UPDATE training_data
                 SET white_wins = white_wins + ?1,
                     black_wins = black_wins + ?2,
                     total_games = total_games + ?3,
                     updated_at = CURRENT_TIMESTAMP
                 WHERE id = ?4",
                [white_wins, black_wins, trials_per_record as i32, id as i32],
            )?;

            updated_count += 1;

            if updated_count % 10 == 0 {
                println!("{}個のレコードを更新しました", updated_count);
            }
        }

        println!("{}個のレコードを更新しました", updated_count);
        Ok(updated_count)
    }

    /// 指定された盤面でランダム対局を実行
    ///
    /// # Arguments
    /// * `board` - 開始盤面
    /// * `trials` - 試行回数
    ///
    /// # Returns
    /// * `(i32, i32)` - (白の勝利数, 黒の勝利数)
    fn run_random_games(&self, board: &Board, trials: usize) -> (i32, i32) {
        let mut white_wins = 0;
        let mut black_wins = 0;

        for _ in 0..trials {
            let mut game = Game::from(board.clone(), 1, ColorType::Black, ColorType::None);
            let result = game.random_play();

            match result.winner {
                ColorType::White => white_wins += 1,
                ColorType::Black => black_wins += 1,
                ColorType::None => {
                    // 引き分けの場合はランダムに勝者を決定
                    if rand::random::<bool>() {
                        white_wins += 1;
                    } else {
                        black_wins += 1;
                    }
                }
                ColorType::ColorNumber => {
                    // このケースは通常発生しないが、安全のため追加
                    if rand::random::<bool>() {
                        white_wins += 1;
                    } else {
                        black_wins += 1;
                    }
                }
            }
        }

        (white_wins, black_wins)
    }

    /// 学習データを取得してモデルを訓練
    ///
    /// # Arguments
    /// * `min_games` - 最小ゲーム数（この数以上のゲームが実行されたレコードのみ使用）
    /// * `training_config` - 学習設定
    /// * `model_save_path` - モデル保存パス
    ///
    /// # Returns
    /// * `SqlResult<()>` - 学習結果
    pub fn train_model(
        &self,
        min_games: i32,
        training_config: TrainingConfig,
        model_save_path: String,
    ) -> SqlResult<()> {
        let conn = self.get_connection()?;

        let mut stmt = conn.prepare(
            "SELECT board, white_wins, black_wins, total_games
             FROM training_data
             WHERE total_games >= ?1
             ORDER BY total_games DESC",
        )?;

        let mut records: Vec<(String, i32, i32, i32)> = Vec::new();
        let rows = stmt.query_map([min_games], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, i32>(2)?,
                row.get::<_, i32>(3)?,
            ))
        })?;

        for row_result in rows {
            let (board_sfen, white_wins, black_wins, total_games) = row_result?;
            records.push((board_sfen, white_wins, black_wins, total_games));
        }

        let mut training_data = TrainingData::new();

        for (board_sfen, white_wins, black_wins, total_games) in records {
            let board = Board::from_sfen(board_sfen);
            let board_vector = board.to_vector(None);

            let white_win_rate = if total_games > 0 {
                white_wins as f32 / total_games as f32
            } else {
                0.5
            };
            let black_win_rate = if total_games > 0 {
                black_wins as f32 / total_games as f32
            } else {
                0.5
            };

            let target = vec![white_win_rate, black_win_rate, total_games as f32];
            training_data.add_sample(board_vector, target);
        }

        if training_data.is_empty() {
            return Err(rusqlite::Error::InvalidParameterName(
                "学習データが見つかりません".to_string(),
            ));
        }

        println!("{}個の学習データを取得しました", training_data.len());

        let device = NdArrayDevice::default();
        let model_config = NnModelConfig::default();
        let model: NnModel<Autodiff<NdArray>> = NnModel::new(&model_config, &device);

        match model.train(&training_data, &training_config, &device) {
            Ok(trained_model) => {
                println!("モデルの訓練が完了しました");

                if let Err(e) = trained_model.save(&model_save_path) {
                    eprintln!("モデルの保存に失敗しました: {}", e);
                } else {
                    println!("モデルを保存しました: {}", model_save_path);
                }
            }
            Err(e) => {
                return Err(rusqlite::Error::InvalidParameterName(format!(
                    "モデルの訓練に失敗しました: {}",
                    e
                )));
            }
        }

        Ok(())
    }

    /// モデルを読み込み、任意の盤面で推論を実行（評価関数実行）
    ///
    /// # Arguments
    /// * `board` - 評価する盤面
    /// * `model_path` - モデルファイルのパス
    ///
    /// # Returns
    /// * `SqlResult<(f32, f32, f32)>` - (白の勝率予測, 黒の勝率予測, 総ゲーム数予測)
    pub fn evaluate_position(&self, board: &Board, model_path: &str) -> SqlResult<(f32, f32, f32)> {
        if !Path::new(model_path).exists() {
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "モデルファイルが見つかりません: {}",
                model_path
            )));
        }

        let device = NdArrayDevice::default();
        let model: NnModel<Autodiff<NdArray>> =
            NnModel::load(model_path, &device).map_err(|e| {
                rusqlite::Error::InvalidParameterName(format!(
                    "モデルの読み込みに失敗しました: {}",
                    e
                ))
            })?;

        let board_vector = board.to_vector(None);
        let prediction = model.predict_single(board_vector);

        let white_win_rate = prediction.clone().slice([0..1]).into_scalar();
        let black_win_rate = prediction.clone().slice([1..2]).into_scalar();
        let total_games = prediction.slice([2..3]).into_scalar();

        Ok((white_win_rate, black_win_rate, total_games))
    }

    /// データベースの統計情報を取得
    ///
    /// # Returns
    /// * `SqlResult<(i32, i32, i32)>` - (総レコード数, 総ゲーム数, 平均ゲーム数)
    pub fn get_database_stats(&self) -> SqlResult<(i32, i32, i32)> {
        let conn = self.get_connection()?;

        let mut stmt = conn.prepare(
            "SELECT COUNT(*), COALESCE(SUM(total_games), 0), COALESCE(AVG(total_games), 0) FROM training_data"
        )?;

        let result = stmt.query_row([], |row| {
            Ok((
                row.get::<_, i32>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, f64>(2)? as i32,
            ))
        })?;

        Ok(result)
    }
}

/// Python用のラッパー関数
#[pymethods]
impl Evaluator {
    #[new]
    pub fn new_for_python(db_path: String) -> Self {
        Self::new(db_path)
    }

    #[pyo3(name = "init_database")]
    pub fn python_init_database(&self) -> PyResult<()> {
        self.init_database()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "generate_and_save_random_boards")]
    pub fn python_generate_and_save_random_boards(&self, count: usize) -> PyResult<i32> {
        self.generate_and_save_random_boards(count)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "update_records_with_random_games")]
    pub fn python_update_records_with_random_games(
        &self,
        trials_per_record: usize,
        max_records: Option<usize>,
    ) -> PyResult<i32> {
        self.update_records_with_random_games(trials_per_record, max_records)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "train_model")]
    pub fn python_train_model(
        &self,
        min_games: i32,
        learning_rate: f64,
        batch_size: usize,
        num_epochs: usize,
        model_save_path: String,
    ) -> PyResult<()> {
        let training_config = TrainingConfig {
            learning_rate,
            batch_size,
            num_epochs,
            model_save_path: model_save_path.clone(),
        };

        self.train_model(min_games, training_config, model_save_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "evaluate_position")]
    pub fn python_evaluate_position(
        &self,
        board: &Board,
        model_path: String,
    ) -> PyResult<(f32, f32, f32)> {
        self.evaluate_position(board, &model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "get_database_stats")]
    pub fn python_get_database_stats(&self) -> PyResult<(i32, i32, i32)> {
        self.get_database_stats()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
