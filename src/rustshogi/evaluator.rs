use super::board::Board;
use super::color::ColorType;
use super::game::Game;
use super::nn_model::{NnModel, NnModelConfig, TrainingConfig, TrainingData};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use pyo3::prelude::*;
use rand;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

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

/// データベースタイプの列挙型
#[derive(Debug, Clone)]
pub enum DatabaseType {
    Sqlite(String),   // SQLiteファイルパス
    Postgres(String), // PostgreSQL接続文字列
}

/// 評価関数システム
#[pyclass]
pub struct Evaluator {
    db_type: DatabaseType,
}

impl Evaluator {
    /// 新しい評価関数システムを作成
    pub fn new(db_type: DatabaseType) -> Self {
        Self { db_type }
    }

    /// データベーステーブルを初期化
    pub fn init_database(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match &self.db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;

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

                println!("SQLiteデータベーステーブルを初期化しました: {}", db_path);
                Ok(())
            }
            DatabaseType::Postgres(connection_string) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let (client, connection) = tokio_postgres::connect(connection_string, tokio_postgres::NoTls).await?;

                    // 接続をバックグラウンドで実行
                    tokio::spawn(async move {
                        if let Err(e) = connection.await {
                            eprintln!("PostgreSQL接続エラー: {}", e);
                        }
                    });

                    client.execute(
                        "CREATE TABLE IF NOT EXISTS training_data (
                            id SERIAL PRIMARY KEY,
                            board TEXT NOT NULL,
                            white_wins INTEGER DEFAULT 0,
                            black_wins INTEGER DEFAULT 0,
                            total_games INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )",
                        &[],
                    ).await?;

                    client.execute(
                        "CREATE INDEX IF NOT EXISTS idx_training_data_total_games ON training_data(total_games)",
                        &[],
                    ).await?;

                    println!("PostgreSQLデータベーステーブルを初期化しました");
                    Ok::<(), tokio_postgres::Error>(())
                }).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }

    /// ランダム盤面を生成してRDBに保存
    ///
    /// # Arguments
    /// * `count` - 生成する盤面の数
    ///
    /// # Returns
    /// * `Result<i32, Box<dyn std::error::Error + Send + Sync>>` - 保存されたレコード数
    pub fn generate_and_save_random_boards(
        &self,
        count: usize,
    ) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
        let mut saved_count = 0;
        let start_time = Instant::now();

        println!("{}個のランダム盤面を生成中...", count);

        for i in 0..count {
            let mut game = Game::new();
            game.input_board("startpos".to_string());
            let random_board = game.generate_random_board();
            let board_sfen = random_board.to_string();

            match &self.db_type {
                DatabaseType::Sqlite(db_path) => {
                    let conn = rusqlite::Connection::open(db_path)?;
                    conn.execute(
                        "INSERT INTO training_data (board) VALUES (?1)",
                        [&board_sfen],
                    )?;
                }
                DatabaseType::Postgres(connection_string) => {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let (client, connection) =
                            tokio_postgres::connect(connection_string, tokio_postgres::NoTls)
                                .await?;

                        tokio::spawn(async move {
                            if let Err(e) = connection.await {
                                eprintln!("PostgreSQL接続エラー: {}", e);
                            }
                        });

                        client
                            .execute(
                                "INSERT INTO training_data (board) VALUES ($1)",
                                &[&board_sfen],
                            )
                            .await?;

                        Ok::<(), tokio_postgres::Error>(())
                    })?;
                }
            }

            saved_count += 1;

            if (i + 1) % 100 == 0 {
                let elapsed = start_time.elapsed();
                println!(
                    "{}個の盤面を生成・保存しました (経過時間: {:.2}秒)",
                    i + 1,
                    elapsed.as_secs_f64()
                );
            }
        }

        let total_elapsed = start_time.elapsed();
        println!(
            "{}個のランダム盤面を生成・保存しました (総時間: {:.2}秒)",
            saved_count,
            total_elapsed.as_secs_f64()
        );
        Ok(saved_count)
    }

    /// 保存されたレコードを読み取り、ランダム対局を実行して勝利数を更新
    ///
    /// # Arguments
    /// * `trials_per_record` - 各レコードに対する試行回数
    /// * `max_records` - 処理する最大レコード数（Noneの場合は全て）
    /// * `num_threads` - スレッド数
    /// * `num_processes` - プロセス数
    ///
    /// # Returns
    /// * `Result<i32, Box<dyn std::error::Error + Send + Sync>>` - 更新されたレコード数
    pub fn update_records_with_random_games(
        &self,
        trials_per_record: usize,
        max_records: Option<usize>,
        num_threads: usize,
        num_processes: usize,
    ) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();
        let records = match &self.db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
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
                records
            }
            DatabaseType::Postgres(connection_string) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let (client, connection) =
                        tokio_postgres::connect(connection_string, tokio_postgres::NoTls).await?;

                    tokio::spawn(async move {
                        if let Err(e) = connection.await {
                            eprintln!("PostgreSQL接続エラー: {}", e);
                        }
                    });

                    let mut query =
                        "SELECT id, board FROM training_data ORDER BY total_games ASC, id ASC"
                            .to_string();
                    if let Some(max) = max_records {
                        query.push_str(&format!(" LIMIT {}", max));
                    }

                    let rows = client.query(&query, &[]).await?;
                    let mut records = Vec::new();

                    for row in rows {
                        let id: i32 = row.get(0);
                        let board: String = row.get(1);
                        records.push((id as i64, board));
                    }
                    Ok::<Vec<(i64, String)>, tokio_postgres::Error>(records)
                })?
            }
        };

        // レコードをスレッド数に応じて分割（プロセス数はスレッド数として使用）
        let chunk_size = if records.is_empty() {
            1 // 空の場合は最小チャンクサイズを設定
        } else {
            records.len().div_ceil(num_processes)
        };
        let record_chunks: Vec<Vec<(i64, String)>> = records
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        println!(
            "{}個のレコードを{}個のスレッドで並列処理します",
            records.len(),
            num_processes
        );

        // 進捗追跡用のアトミックカウンター
        let processed_count = Arc::new(AtomicUsize::new(0));
        let total_records = records.len();

        // 各スレッドで処理する関数
        let processed_count_clone = processed_count.clone();
        let process_records = move |chunk: Vec<(i64, String)>| -> Result<
            Vec<(i64, i32, i32, i32)>,
            Box<dyn std::error::Error + Send + Sync>,
        > {
            let mut results = Vec::new();

            for (id, board_sfen) in chunk {
                // SFEN形式からBoardを復元
                let board = Board::from_sfen(board_sfen);
                let color = if rand::random::<bool>() {
                    ColorType::White
                } else {
                    ColorType::Black
                };
                let game = Game::from(board.clone(), 1, color, ColorType::None);
                let mcts_results = game.random_move_parallel(trials_per_record, num_threads);
                let white_wins = mcts_results.iter().map(|r| r.white_wins as i32).sum();
                let black_wins = mcts_results.iter().map(|r| r.black_wins as i32).sum();
                let total_games = mcts_results.iter().map(|r| r.total_games as i32).sum();

                results.push((id, white_wins, black_wins, total_games));

                // 進捗を更新
                let current_count = processed_count_clone.fetch_add(1, Ordering::Relaxed) + 1;

                // 10個ごと、または最後のレコードで進捗を表示
                if current_count % 10 == 0 || current_count == total_records {
                    let elapsed = start_time.elapsed();
                    let progress_percent = (current_count as f64 / total_records as f64) * 100.0;
                    println!(
                        "進捗: {}/{} ({:.1}%) - 経過時間: {:.2}秒",
                        current_count,
                        total_records,
                        progress_percent,
                        elapsed.as_secs_f64()
                    );
                }
            }

            Ok(results)
        };

        // マルチスレッドで並列実行（スレッドプールを設定）
        let all_results: Result<Vec<_>, _> = rayon::ThreadPoolBuilder::new()
            .num_threads(num_processes)
            .build()
            .unwrap()
            .install(|| record_chunks.into_par_iter().map(process_records).collect());

        let all_results = all_results?;
        println!(
            "ランダム対局の試行が完了しました。結果チャンク数: {}",
            all_results.len()
        );

        // データベース書き込みも並列化
        println!("データベースへの書き込みを開始します...");
        let db_type_clone = self.db_type.clone();
        let write_results: Result<Vec<i32>, Box<dyn std::error::Error + Send + Sync>> = all_results
            .into_par_iter()
            .map(
                |results| -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
                    let mut local_count = 0;
                    for (id, white_wins, black_wins, total_games) in results {
                        match &db_type_clone {
                            DatabaseType::Sqlite(db_path) => {
                                let conn = rusqlite::Connection::open(db_path)?;
                                conn.execute(
                                    "UPDATE training_data
                                 SET white_wins = white_wins + ?1,
                                     black_wins = black_wins + ?2,
                                     total_games = total_games + ?3,
                                     updated_at = CURRENT_TIMESTAMP
                                 WHERE id = ?4",
                                    [white_wins, black_wins, total_games, id as i32],
                                )?;
                            }
                            DatabaseType::Postgres(connection_string) => {
                                let rt = tokio::runtime::Runtime::new().unwrap();
                                rt.block_on(async {
                                    let (client, connection) = tokio_postgres::connect(
                                        connection_string,
                                        tokio_postgres::NoTls,
                                    )
                                    .await?;

                                    tokio::spawn(async move {
                                        if let Err(e) = connection.await {
                                            eprintln!("PostgreSQL接続エラー: {}", e);
                                        }
                                    });

                                    client
                                        .execute(
                                            "UPDATE training_data
                                     SET white_wins = white_wins + $1,
                                         black_wins = black_wins + $2,
                                         total_games = total_games + $3,
                                         updated_at = CURRENT_TIMESTAMP
                                     WHERE id = $4",
                                            &[&white_wins, &black_wins, &total_games, &(id as i32)],
                                        )
                                        .await?;

                                    Ok::<(), tokio_postgres::Error>(())
                                })?;
                            }
                        }
                        local_count += 1;
                    }
                    Ok(local_count)
                },
            )
            .collect();

        let write_counts = write_results?;
        let updated_count = write_counts.iter().sum();

        println!(
            "データベースへの書き込みが完了しました。更新されたレコード数: {}",
            updated_count
        );

        let total_elapsed = start_time.elapsed();
        println!(
            "{}個のレコードを更新しました (総時間: {:.2}秒)",
            updated_count,
            total_elapsed.as_secs_f64()
        );
        Ok(updated_count)
    }

    /// 学習データを取得してモデルを訓練
    ///
    /// # Arguments
    /// * `min_games` - 最小ゲーム数（この数以上のゲームが実行されたレコードのみ使用）
    /// * `training_config` - 学習設定
    /// * `model_save_path` - モデル保存パス
    ///
    /// # Returns
    /// * `Result<(), Box<dyn std::error::Error + Send + Sync>>` - 学習結果
    pub fn train_model(
        &self,
        min_games: i32,
        training_config: TrainingConfig,
        model_save_path: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let records = match &self.db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
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
                records
            }
            DatabaseType::Postgres(connection_string) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let (client, connection) =
                        tokio_postgres::connect(connection_string, tokio_postgres::NoTls).await?;

                    tokio::spawn(async move {
                        if let Err(e) = connection.await {
                            eprintln!("PostgreSQL接続エラー: {}", e);
                        }
                    });

                    let rows = client
                        .query(
                            "SELECT board, white_wins, black_wins, total_games
                         FROM training_data
                         WHERE total_games >= $1
                         ORDER BY total_games DESC",
                            &[&min_games],
                        )
                        .await?;

                    let mut records = Vec::new();
                    for row in rows {
                        let board: String = row.get(0);
                        let white_wins: i32 = row.get(1);
                        let black_wins: i32 = row.get(2);
                        let total_games: i32 = row.get(3);
                        records.push((board, white_wins, black_wins, total_games));
                    }
                    Ok::<Vec<(String, i32, i32, i32)>, tokio_postgres::Error>(records)
                })?
            }
        };

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
            return Err("学習データが見つかりません".into());
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
                return Err(format!("モデルの訓練に失敗しました: {}", e).into());
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
    /// * `Result<(f32, f32, f32), Box<dyn std::error::Error + Send + Sync>>` - (白の勝率予測, 黒の勝率予測, 総ゲーム数予測)
    pub fn evaluate_position(
        &self,
        board: &Board,
        model_path: &str,
    ) -> Result<(f32, f32, f32), Box<dyn std::error::Error + Send + Sync>> {
        if !Path::new(model_path).exists() {
            return Err(format!("モデルファイルが見つかりません: {}", model_path).into());
        }

        let device = NdArrayDevice::default();
        let model: NnModel<Autodiff<NdArray>> = NnModel::load(model_path, &device).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                format!("モデルの読み込みに失敗しました: {}", e).into()
            },
        )?;

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
    /// * `Result<(i32, i32, i32), Box<dyn std::error::Error + Send + Sync>>` - (総レコード数, 総ゲーム数, 平均ゲーム数)
    pub fn get_database_stats(
        &self,
    ) -> Result<(i32, i32, i32), Box<dyn std::error::Error + Send + Sync>> {
        match &self.db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
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
            DatabaseType::Postgres(connection_string) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let (client, connection) = tokio_postgres::connect(connection_string, tokio_postgres::NoTls).await?;

                    tokio::spawn(async move {
                        if let Err(e) = connection.await {
                            eprintln!("PostgreSQL接続エラー: {}", e);
                        }
                    });

                    let row = client.query_one(
                        "SELECT COUNT(*), COALESCE(SUM(total_games), 0), COALESCE(CAST(AVG(total_games) AS DOUBLE PRECISION), 0) FROM training_data",
                        &[],
                    ).await?;

                    let count: i64 = row.get(0);
                    let total_games: i64 = row.get(1);
                    let avg_games: f64 = row.get(2);

                    Ok::<(i32, i32, i32), tokio_postgres::Error>((count as i32, total_games as i32, avg_games as i32))
                }).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }
}

/// Python用のラッパー関数
#[pymethods]
impl Evaluator {
    #[new]
    pub fn new_for_python(db_type_str: String, connection_string: String) -> Self {
        let db_type = if db_type_str.to_lowercase() == "postgres"
            || db_type_str.to_lowercase() == "postgresql"
        {
            DatabaseType::Postgres(connection_string)
        } else {
            DatabaseType::Sqlite(connection_string)
        };
        Self::new(db_type)
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
        num_threads: usize,
        num_processes: usize,
    ) -> PyResult<i32> {
        self.update_records_with_random_games(
            trials_per_record,
            max_records,
            num_threads,
            num_processes,
        )
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
