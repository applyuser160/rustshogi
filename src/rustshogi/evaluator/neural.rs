use super::super::board::Board;
use super::super::color::ColorType;
use super::super::game::Game;
use super::super::nn_model::{NnModel, NnModelConfig, TrainingConfig, TrainingData};
use super::abst::Evaluator;
use super::simple::SimpleEvaluator;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use pyo3::prelude::*;
use rand;
use serde::{Deserialize, Serialize};
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

    /// 学習データを取得してモデルを訓練
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
        let records = match db_type {
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

        let records = if let Some(max_samples) = max_samples {
            if records.len() > max_samples {
                println!(
                    "データをサンプリング中... ({}個から{}個に削減)",
                    records.len(),
                    max_samples
                );

                use rand::seq::SliceRandom;
                use rand::thread_rng;

                let mut rng = thread_rng();
                let mut sampled_records = Vec::with_capacity(max_samples);

                let mut weighted_indices: Vec<(usize, i32)> = records
                    .iter()
                    .enumerate()
                    .map(|(i, (_, _, _, total_games))| (i, *total_games))
                    .collect();

                for _ in 0..max_samples {
                    if let Ok(&(idx, _)) = weighted_indices.choose_weighted(&mut rng, |item| item.1)
                    {
                        sampled_records.push(records[idx].clone());
                        weighted_indices.retain(|(i, _)| *i != idx);
                    }
                }

                println!(
                    "サンプリング完了: {}個のレコードを選択",
                    sampled_records.len()
                );
                sampled_records
            } else {
                records
            }
        } else {
            records
        };

        let mut training_data = TrainingData::new();
        training_data.inputs.reserve(records.len());
        training_data.targets.reserve(records.len());

        println!("{}個のレコードを処理中...", records.len());
        let processing_start = Instant::now();

        for (i, (board_sfen, white_wins, black_wins, total_games)) in records.iter().enumerate() {
            let board = Board::from_sfen(board_sfen.clone());
            let board_vector = board.to_vector(None);

            let white_win_rate = if *total_games > 0 {
                *white_wins as f32 / *total_games as f32
            } else {
                0.5
            };
            let black_win_rate = if *total_games > 0 {
                *black_wins as f32 / *total_games as f32
            } else {
                0.5
            };
            let draw_rate = if *total_games > 0 {
                (*total_games - *white_wins - *black_wins) as f32 / *total_games as f32
            } else {
                0.0
            };

            let target = vec![white_win_rate, black_win_rate, draw_rate];
            training_data.add_sample(board_vector, target);

            if (i + 1) % 1000 == 0 {
                let elapsed = processing_start.elapsed();
                println!(
                    "処理済み: {}/{} ({:.1}%) - 経過時間: {:.2}秒",
                    i + 1,
                    records.len(),
                    (i + 1) as f64 / records.len() as f64 * 100.0,
                    elapsed.as_secs_f64()
                );
            }
        }

        if training_data.is_empty() {
            return Err("学習データが見つかりません".into());
        }

        println!("{}個の学習データを取得しました", training_data.len());

        let device = NdArrayDevice::default();
        let model_config = NnModelConfig::default();

        let model: NnModel<Autodiff<NdArray>>;
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

        match model.train(&training_data, &training_config, &device) {
            Ok(trained_model) => {
                let training_elapsed = start_time.elapsed();
                println!(
                    "モデルの訓練が完了しました (訓練時間: {:.2}秒)",
                    training_elapsed.as_secs_f64()
                );

                let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

                if let Err(e) = trained_model.clone().save_file(&model_save_path, &recorder) {
                    eprintln!("モデルの保存に失敗しました: {}", e);
                } else {
                    println!("モデルを保存しました: {}", &model_save_path);
                }
            }
            Err(e) => {
                return Err(format!("モデルの訓練に失敗しました: {}", e).into());
            }
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
    ) -> Result<(i32, i32, i32), Box<dyn std::error::Error + Send + Sync>> {
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
    pub fn python_get_database_stats(&self) -> PyResult<(i32, i32, i32)> {
        self.get_database_stats()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "get_record_details")]
    pub fn python_get_record_details(&self, record_id: i64) -> PyResult<Option<TrainingRecord>> {
        self.get_record_details(record_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
