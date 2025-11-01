use super::super::board::Board;
use super::super::color::ColorType;
use super::super::game::Game;
use super::super::nn_model::{NnModel, NnModelConfig, TrainingConfig};
use super::abst::Evaluator;
use super::simple::SimpleEvaluator;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use chrono;
use pyo3::prelude::*;
use rand;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

/// 学習データベースのレコード構造体
#[derive(Debug, Serialize, Deserialize)]
#[pyclass]
pub struct TrainingRecord {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub board: String, // SFEN形式の文字列
    #[pyo3(get)]
    pub white_wins: i32,
    #[pyo3(get)]
    pub black_wins: i32,
    #[pyo3(get)]
    pub total_games: i32,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
}

/// データベースタイプの列挙型
#[derive(Debug, Clone)]
pub enum DatabaseType {
    Sqlite(String),   // SQLiteファイルパス
    Postgres(String), // PostgreSQL接続文字列
}

/// ニューラルネットワーク評価関数システム
#[pyclass]
#[derive(Clone)]
pub struct NeuralEvaluator {
    db_type: Option<DatabaseType>,
    model_path: Option<String>,
}

impl NeuralEvaluator {
    /// 新しい評価関数システムを作成
    pub fn new(db_type: Option<DatabaseType>, model_path: Option<String>) -> Self {
        Self {
            db_type,
            model_path,
        }
    }

    /// データベーステーブルを初期化
    pub fn init_database(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db_type = self
            .db_type
            .as_ref()
            .ok_or("データベースタイプが設定されていません")?;
        match db_type {
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
    pub fn generate_and_save_random_boards(
        &self,
        count: usize,
    ) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
        let db_type = self
            .db_type
            .as_ref()
            .ok_or("データベースタイプが設定されていません")?;
        let mut saved_count = 0;
        let start_time = Instant::now();

        println!("{}個のランダム盤面を生成中...", count);

        match db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;

                for i in 0..count {
                    let mut game = Game::new();
                    game.input_board("startpos".to_string());
                    let random_board = game.generate_random_board();
                    let board_sfen = random_board.to_string();

                    conn.execute(
                        "INSERT INTO training_data (board) VALUES (?1)",
                        [&board_sfen],
                    )?;

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

                    for i in 0..count {
                        let mut game = Game::new();
                        game.input_board("startpos".to_string());
                        let random_board = game.generate_random_board();
                        let board_sfen = random_board.to_string();

                        client
                            .execute(
                                "INSERT INTO training_data (board) VALUES ($1)",
                                &[&board_sfen],
                            )
                            .await?;

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

                    Ok::<(), tokio_postgres::Error>(())
                })?;
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
    pub fn update_records_with_random_games(
        &self,
        trials_per_record: usize,
        max_records: Option<usize>,
        num_threads: usize,
    ) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
        let db_type = self
            .db_type
            .as_ref()
            .ok_or("データベースタイプが設定されていません")?;
        let start_time = Instant::now();
        let records = match db_type {
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

        println!(
            "{}個のレコードを順次処理します（各レコード内で{}個のスレッドで並列実行）",
            records.len(),
            num_threads
        );

        match self.get_database_stats() {
            Ok((total_records, total_games, avg_games)) => {
                println!(
                    "処理開始前のデータベース統計: 総レコード数={}, 総ゲーム数={}, 平均ゲーム数={}",
                    total_records, total_games, avg_games
                );
            }
            Err(e) => {
                eprintln!("データベース統計の取得に失敗: {}", e);
            }
        }

        let mut processed_count = 0;
        let total_records = records.len();
        let mut all_results = Vec::new();

        for (id, board_sfen) in records {
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

            all_results.push((id, white_wins, black_wins, total_games));
            processed_count += 1;

            if processed_count % 10 == 0 || processed_count == total_records {
                let elapsed = start_time.elapsed();
                let progress_percent = (processed_count as f64 / total_records as f64) * 100.0;
                println!(
                    "進捗: {}/{} ({:.1}%) - 経過時間: {:.2}秒",
                    processed_count,
                    total_records,
                    progress_percent,
                    elapsed.as_secs_f64()
                );
            }
        }

        println!(
            "ランダム対局の試行が完了しました。処理されたレコード数: {}",
            all_results.len()
        );

        println!("データベースへの書き込みを開始します...");
        let mut updated_count = 0;

        for (id, white_wins, black_wins, total_games) in all_results {
            let update_result = match db_type {
                DatabaseType::Sqlite(db_path) => {
                    let conn = rusqlite::Connection::open(db_path)?;
                    let rows_affected = conn.execute(
                        "UPDATE training_data
                         SET white_wins = white_wins + ?1,
                             black_wins = black_wins + ?2,
                             total_games = total_games + ?3,
                             updated_at = CURRENT_TIMESTAMP
                         WHERE id = ?4",
                        [white_wins, black_wins, total_games, id as i32],
                    )?;

                    if rows_affected == 0 {
                        eprintln!(
                            "警告: レコードID {} の更新に失敗しました（レコードが見つかりません）",
                            id
                        );
                        false
                    } else {
                        println!(
                            "レコードID {} を更新しました (白勝: +{}, 黒勝: +{}, 総ゲーム: +{})",
                            id, white_wins, black_wins, total_games
                        );
                        true
                    }
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

                        let rows_affected = client
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

                        if rows_affected == 0 {
                            eprintln!("警告: レコードID {} の更新に失敗しました（レコードが見つかりません）", id);
                            Ok(false)
                        } else {
                            Ok(true)
                        }
                    }).map_err(|e: tokio_postgres::Error| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?
                }
            };

            if update_result {
                updated_count += 1;
            }
        }

        println!(
            "データベースへの書き込みが完了しました。更新されたレコード数: {}",
            updated_count
        );

        match self.get_database_stats() {
            Ok((total_records, total_games, avg_games)) => {
                println!(
                    "データベース統計: 総レコード数={}, 総ゲーム数={}, 平均ゲーム数={}",
                    total_records, total_games, avg_games
                );
            }
            Err(e) => {
                eprintln!("データベース統計の取得に失敗: {}", e);
            }
        }

        let total_elapsed = start_time.elapsed();
        println!(
            "{}個のレコードを更新しました (総時間: {:.2}秒)",
            updated_count,
            total_elapsed.as_secs_f64()
        );
        Ok(updated_count)
    }

    /// 損失関数（AutodiffBackend用）
    fn mse_loss_autodiff<B: AutodiffBackend>(
        predictions: &Tensor<B, 2>,
        targets: &Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let diff = predictions.clone() - targets.clone();
        let squared_diff = diff.clone() * diff;
        squared_diff.mean()
    }

    /// データベースからバッチ単位でデータを取得するヘルパー関数
    fn fetch_batch_from_db(
        &self,
        db_type: &DatabaseType,
        min_games: i32,
        batch_size: usize,
        offset: usize,
    ) -> Result<Vec<(String, i32, i32, i32)>, Box<dyn std::error::Error + Send + Sync>> {
        match db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
                let mut stmt = conn.prepare(
                    "SELECT board, white_wins, black_wins, total_games
                     FROM training_data
                     WHERE total_games >= ?1
                     ORDER BY total_games DESC
                     LIMIT ?2 OFFSET ?3",
                )?;

                let mut batch: Vec<(String, i32, i32, i32)> = Vec::new();
                let limit_param = batch_size as i32;
                let offset_param = offset as i32;
                let rows = stmt.query_map(
                    rusqlite::params![min_games, limit_param, offset_param],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i32>(1)?,
                            row.get::<_, i32>(2)?,
                            row.get::<_, i32>(3)?,
                        ))
                    },
                )?;

                for row_result in rows {
                    batch.push(row_result?);
                }
                Ok(batch)
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
                            &format!(
                                "SELECT board, white_wins, black_wins, total_games
                                 FROM training_data
                                 WHERE total_games >= $1
                                 ORDER BY total_games DESC
                                 LIMIT {} OFFSET {}",
                                batch_size, offset
                            ),
                            &[&min_games],
                        )
                        .await?;

                    let mut batch = Vec::new();
                    for row in rows {
                        let board: String = row.get(0);
                        let white_wins: i32 = row.get(1);
                        let black_wins: i32 = row.get(2);
                        let total_games: i32 = row.get(3);
                        batch.push((board, white_wins, black_wins, total_games));
                    }
                    Ok::<Vec<(String, i32, i32, i32)>, tokio_postgres::Error>(batch)
                })
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }

    /// 学習データを取得してモデルを訓練（ストリーミング処理で全データを使用）
    pub fn train_model(
        &self,
        min_games: i32,
        training_config: TrainingConfig,
        model_save_path: String,
        max_samples: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db_type = self
            .db_type
            .as_ref()
            .ok_or("データベースタイプが設定されていません")?;
        let start_time = Instant::now();
        println!("モデルの訓練を開始します...");

        // まず総レコード数を取得
        let total_count = match db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
                let count: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM training_data WHERE total_games >= ?1",
                    [min_games],
                    |row| row.get(0),
                )?;
                count as usize
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

                    let row = client
                        .query_one(
                            "SELECT COUNT(*) FROM training_data WHERE total_games >= $1",
                            &[&min_games],
                        )
                        .await?;
                    let count: i64 = row.get(0);
                    Ok::<usize, tokio_postgres::Error>(count as usize)
                })?
            }
        };

        // max_samplesが指定されていない場合は全データを使用（ストリーミング処理でメモリ効率的）
        let target_count = max_samples.unwrap_or(total_count).min(total_count);
        println!(
            "総レコード数: {} (使用: {} - ストリーミング処理で全データを使用)",
            total_count, target_count
        );

        // 学習データが存在しない場合はエラーを返す
        if target_count == 0 {
            return Err("学習データが見つかりません".into());
        }

        // データベースからの読み込みバッチサイズ（メモリ効率化）
        const DB_FETCH_BATCH_SIZE: usize = 1000;

        // ストリーミング処理用: モデルとオプティマイザーを初期化
        let device = NdArrayDevice::default();
        let model_config = NnModelConfig::default();

        let mut model: NnModel<Autodiff<NdArray>>;
        if Path::new(&model_save_path).exists() {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            model = NnModel::new(&model_config, &device).load_file(
                &model_save_path,
                &recorder,
                &device,
            )?;
        } else {
            model = NnModel::new(&model_config, &device);
        }

        let optim_config = AdamConfig::new();
        let mut optim = optim_config.init();

        // 学習パラメータ
        let input_dim = 2320;
        let output_dim = 3;
        let batch_size = training_config.batch_size;

        println!("ストリーミング処理で学習を開始します（全データを使用）...");
        println!(
            "エポック数: {}, バッチサイズ: {}",
            training_config.num_epochs, batch_size
        );

        // 早期停止用の変数
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        // エポックごとの学習ループ
        for epoch in 0..training_config.num_epochs {
            let epoch_start_time = Instant::now();
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            // 学習率スケジューリング
            let current_lr = if training_config.use_lr_scheduling {
                training_config.learning_rate * (0.95_f64.powi(epoch as i32))
            } else {
                training_config.learning_rate
            };

            println!(
                "エポック {} 開始: データベースからストリーミング読み込み...",
                epoch
            );

            // データベースからストリーミング読み込み
            let mut db_offset = 0;
            let mut processed_samples = 0;
            let progress_update_interval = 1000; // 1000サンプルごとに進捗表示

            loop {
                // データベースからバッチを取得
                let batch_records =
                    self.fetch_batch_from_db(db_type, min_games, DB_FETCH_BATCH_SIZE, db_offset)?;

                if batch_records.is_empty() {
                    break;
                }

                // バッチ内のデータを処理して学習バッチを作成
                let mut batch_inputs = Vec::new();
                let mut batch_targets = Vec::new();

                for (board_sfen, white_wins, black_wins, total_games) in batch_records {
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
                    let draw_rate = if total_games > 0 {
                        (total_games - white_wins - black_wins) as f32 / total_games as f32
                    } else {
                        0.0
                    };

                    batch_inputs.push(board_vector);
                    batch_targets.push(vec![white_win_rate, black_win_rate, draw_rate]);
                }

                // 学習バッチサイズで分割して処理
                for batch_start in (0..batch_inputs.len()).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(batch_inputs.len());
                    let current_batch_size = batch_end - batch_start;

                    // バッチデータを準備
                    let mut flat_inputs = Vec::with_capacity(current_batch_size * input_dim);
                    let mut flat_targets = Vec::with_capacity(current_batch_size * output_dim);

                    for i in batch_start..batch_end {
                        flat_inputs.extend_from_slice(&batch_inputs[i]);
                        flat_targets.extend_from_slice(&batch_targets[i]);
                    }

                    // テンソルに変換
                    let batch_input_tensor = Tensor::<Autodiff<NdArray>, 1>::from_floats(
                        flat_inputs.as_slice(),
                        &device,
                    )
                    .reshape([current_batch_size, input_dim]);

                    let batch_target_tensor = Tensor::<Autodiff<NdArray>, 1>::from_floats(
                        flat_targets.as_slice(),
                        &device,
                    )
                    .reshape([current_batch_size, output_dim]);

                    // フォワードパス
                    let predictions = model.forward(batch_input_tensor);

                    // 損失計算
                    let loss = Self::mse_loss_autodiff(&predictions, &batch_target_tensor);
                    let loss_value: f32 = loss.clone().into_scalar();
                    total_loss += loss_value;

                    // バックプロパゲーション
                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optim.step(current_lr, model, grads_params);

                    batch_count += 1;
                    processed_samples += current_batch_size;

                    // 進捗表示（一定間隔ごと、または最後のバッチ）
                    if processed_samples % progress_update_interval == 0
                        || processed_samples >= target_count
                        || (db_offset + DB_FETCH_BATCH_SIZE >= target_count
                            && batch_start + batch_size >= batch_inputs.len())
                    {
                        let progress_percent =
                            (processed_samples as f64 / target_count as f64) * 100.0;
                        let elapsed = epoch_start_time.elapsed();
                        let elapsed_secs = elapsed.as_secs_f64();

                        // 残り時間の計算
                        let remaining_samples = target_count - processed_samples;
                        let samples_per_sec = if elapsed_secs > 0.0 {
                            processed_samples as f64 / elapsed_secs
                        } else {
                            0.0
                        };
                        let estimated_remaining_secs = if samples_per_sec > 0.0 {
                            remaining_samples as f64 / samples_per_sec
                        } else {
                            0.0
                        };
                        let estimated_remaining_mins = estimated_remaining_secs / 60.0;

                        // 現在時刻から予想終了時刻を計算
                        let now = std::time::SystemTime::now();
                        let estimated_end =
                            now + std::time::Duration::from_secs(estimated_remaining_secs as u64);
                        let end_time_str = chrono::DateTime::<chrono::Local>::from(estimated_end)
                            .format("%H:%M:%S")
                            .to_string();

                        // 同じ行を上書きするために\rを使う
                        print!(
                            "\r  進捗: {}/{} ({:.1}%) - 経過: {:.1}秒 - 速度: {:.0} サンプル/秒 - 残り: {:.1}分 (予想終了: {})    ",
                            processed_samples,
                            target_count,
                            progress_percent,
                            elapsed_secs,
                            samples_per_sec,
                            estimated_remaining_mins,
                            end_time_str
                        );
                        io::stdout().flush().unwrap();
                    }

                    // メモリ解放
                    drop(flat_inputs);
                    drop(flat_targets);
                }

                db_offset += DB_FETCH_BATCH_SIZE;

                if db_offset >= target_count {
                    break;
                }
            }

            // 進捗表示の最後の行をクリア（改行を追加）
            println!();

            let avg_loss = total_loss / batch_count as f32;
            let epoch_elapsed = epoch_start_time.elapsed();
            println!(
                "エポック {} 完了: 平均損失 = {:.6}, バッチ数 = {}, 経過時間 = {:.2}秒",
                epoch,
                avg_loss,
                batch_count,
                epoch_elapsed.as_secs_f64()
            );

            // 早期停止チェック
            if training_config.use_early_stopping {
                if avg_loss < best_loss {
                    best_loss = avg_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= training_config.early_stopping_patience {
                        println!(
                            "早期停止: エポック {} で学習を終了します（損失改善なし）",
                            epoch
                        );
                        break;
                    }
                }
            }
        }

        println!("学習が完了しました");

        // モデルを保存
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        if let Err(e) = model.save_file(&model_save_path, &recorder) {
            eprintln!("モデルの保存に失敗しました: {}", e);
        } else {
            println!("モデルを保存しました: {}", &model_save_path);
        }

        let total_elapsed = start_time.elapsed();
        println!("モデル訓練の総時間: {:.2}秒", total_elapsed.as_secs_f64());
        Ok(())
    }

    /// モデルを読み込み、任意の盤面で推論を実行（評価関数実行）
    pub fn evaluate_position(
        &self,
        board: &Board,
        model_path: Option<&str>,
    ) -> Result<(f32, f32, f32), Box<dyn std::error::Error + Send + Sync>> {
        let path = match model_path {
            Some(p) => p,
            None => self
                .model_path
                .as_ref()
                .ok_or("モデルパスが設定されていません")?
                .as_str(),
        };
        let device = NdArrayDevice::default();
        let model_config = NnModelConfig::default();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model: NnModel<Autodiff<NdArray>> =
            NnModel::new(&model_config, &device).load_file(path, &recorder, &device)?;

        let board_vector = board.to_vector(None);
        let prediction = model.predict_single(board_vector);

        let white_win_rate = prediction.clone().slice([0..1]).into_scalar();
        let black_win_rate = prediction.clone().slice([1..2]).into_scalar();
        let draw_rate = prediction.slice([2..3]).into_scalar();

        Ok((white_win_rate, black_win_rate, draw_rate))
    }

    /// 特定のレコードの詳細情報を取得
    pub fn get_record_details(
        &self,
        record_id: i64,
    ) -> Result<Option<TrainingRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let db_type = self
            .db_type
            .as_ref()
            .ok_or("データベースタイプが設定されていません")?;
        match db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
                let mut stmt = conn.prepare(
                    "SELECT id, board, white_wins, black_wins, total_games, created_at, updated_at
                     FROM training_data WHERE id = ?1",
                )?;

                let result = stmt.query_row([record_id], |row| {
                    Ok(TrainingRecord {
                        id: row.get(0)?,
                        board: row.get(1)?,
                        white_wins: row.get(2)?,
                        black_wins: row.get(3)?,
                        total_games: row.get(4)?,
                        created_at: row.get(5)?,
                        updated_at: row.get(6)?,
                    })
                });

                match result {
                    Ok(record) => Ok(Some(record)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(e.into()),
                }
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

                    let rows = client.query(
                        "SELECT id, board, white_wins, black_wins, total_games, created_at, updated_at
                         FROM training_data WHERE id = $1",
                        &[&record_id],
                    ).await?;

                    if rows.is_empty() {
                        Ok(None)
                    } else {
                        let row = &rows[0];
                        Ok(Some(TrainingRecord {
                            id: row.get(0),
                            board: row.get(1),
                            white_wins: row.get(2),
                            black_wins: row.get(3),
                            total_games: row.get(4),
                            created_at: row.get(5),
                            updated_at: row.get(6),
                        }))
                    }
                }).map_err(|e: tokio_postgres::Error| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }

    /// データベースの統計情報を取得
    pub fn get_database_stats(
        &self,
    ) -> Result<(i64, i64, i64), Box<dyn std::error::Error + Send + Sync>> {
        let db_type = self
            .db_type
            .as_ref()
            .ok_or("データベースタイプが設定されていません")?;
        match db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
                let mut stmt = conn.prepare(
                    "SELECT COUNT(*), COALESCE(SUM(total_games), 0), COALESCE(AVG(total_games), 0) FROM training_data"
                )?;

                let result = stmt.query_row([], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, f64>(2)? as i64,
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

                    Ok::<(i64, i64, i64), tokio_postgres::Error>((count, total_games, avg_games as i64))
                }).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }
}

impl Evaluator for NeuralEvaluator {
    fn evaluate(&self, board: &Board, color: ColorType) -> f32 {
        let model_path = self.model_path.as_deref();
        match self.evaluate_position(board, model_path) {
            Ok((white_win_rate, black_win_rate, _draw_rate)) => match color {
                ColorType::White => (white_win_rate - black_win_rate) * 10000.0,
                ColorType::Black => (black_win_rate - white_win_rate) * 10000.0,
                _ => 0.0,
            },
            Err(_) => {
                let simple_eval = SimpleEvaluator::new();
                simple_eval.evaluate(board, color)
            }
        }
    }
}

#[pymethods]
impl NeuralEvaluator {
    #[new]
    #[pyo3(signature = (db_type_str=None, connection_string=None, model_path=None))]
    pub fn new_for_python(
        db_type_str: Option<String>,
        connection_string: Option<String>,
        model_path: Option<String>,
    ) -> Self {
        let db_type = match (db_type_str, connection_string) {
            (Some(db_str), Some(conn_str)) => {
                if db_str.to_lowercase() == "postgres" || db_str.to_lowercase() == "postgresql" {
                    Some(DatabaseType::Postgres(conn_str))
                } else {
                    Some(DatabaseType::Sqlite(conn_str))
                }
            }
            _ => None,
        };
        Self::new(db_type, model_path)
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
    ) -> PyResult<i32> {
        self.update_records_with_random_games(trials_per_record, max_records, num_threads)
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
            use_lr_scheduling: true,
            use_early_stopping: true,
            early_stopping_patience: 10,
        };

        self.train_model(min_games, training_config, model_save_path, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "train_model_with_sampling")]
    pub fn python_train_model_with_sampling(
        &self,
        min_games: i32,
        learning_rate: f64,
        batch_size: usize,
        num_epochs: usize,
        model_save_path: String,
        max_samples: Option<usize>,
    ) -> PyResult<()> {
        let training_config = TrainingConfig {
            learning_rate,
            batch_size,
            num_epochs,
            model_save_path: model_save_path.clone(),
            use_lr_scheduling: true,
            use_early_stopping: true,
            early_stopping_patience: 10,
        };

        self.train_model(min_games, training_config, model_save_path, max_samples)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "train_model_advanced")]
    pub fn python_train_model_advanced(
        &self,
        min_games: i32,
        learning_rate: f64,
        batch_size: usize,
        num_epochs: usize,
        model_save_path: String,
        use_lr_scheduling: bool,
        use_early_stopping: bool,
        early_stopping_patience: usize,
    ) -> PyResult<()> {
        let training_config = TrainingConfig {
            learning_rate,
            batch_size,
            num_epochs,
            model_save_path: model_save_path.clone(),
            use_lr_scheduling,
            use_early_stopping,
            early_stopping_patience,
        };

        self.train_model(min_games, training_config, model_save_path, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "evaluate_position")]
    #[pyo3(signature = (board, model_path=None))]
    pub fn python_evaluate_position(
        &self,
        board: &Board,
        model_path: Option<String>,
    ) -> PyResult<(f32, f32, f32)> {
        self.evaluate_position(board, model_path.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "get_database_stats")]
    pub fn python_get_database_stats(&self) -> PyResult<(i64, i64, i64)> {
        self.get_database_stats()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "get_record_details")]
    pub fn python_get_record_details(&self, record_id: i64) -> PyResult<Option<TrainingRecord>> {
        self.get_record_details(record_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "evaluate")]
    pub fn python_evaluate(&self, board: &Board, color: ColorType) -> f32 {
        self.evaluate(board, color)
    }
}
