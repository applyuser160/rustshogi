use chrono;
use pyo3::prelude::*;
use rusqlite;
use serde::{Deserialize, Serialize};
use tokio_postgres;

/// 学習データベースのレコード構造体
#[derive(Debug, Serialize, Deserialize, Clone)]
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

/// データベース操作を管理する構造体
pub struct TrainingDatabase {
    pub db_type: DatabaseType,
}

impl TrainingDatabase {
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

    /// データベースからバッチ単位でデータを取得するヘルパー関数
    pub fn fetch_batch_from_db(
        &self,
        min_games: i32,
        batch_size: usize,
        offset: usize,
    ) -> Result<Vec<(String, i32, i32, i32)>, Box<dyn std::error::Error + Send + Sync>> {
        match &self.db_type {
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

    /// 特定のレコードの詳細情報を取得
    pub fn get_record_details(
        &self,
        record_id: i64,
    ) -> Result<Option<TrainingRecord>, Box<dyn std::error::Error + Send + Sync>> {
        match &self.db_type {
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
                        &[&(record_id as i32)],
                    ).await?;

                    if rows.is_empty() {
                        Ok(None)
                    } else {
                        let row = &rows[0];
                        Ok(Some(TrainingRecord {
                            id: row.get::<_, i32>(0) as i64,
                            board: row.get(1),
                            white_wins: row.get(2),
                            black_wins: row.get(3),
                            total_games: row.get(4),
                            created_at: row.get::<_, chrono::NaiveDateTime>(5).to_string(),
                            updated_at: row.get::<_, chrono::NaiveDateTime>(6).to_string(),
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
        match &self.db_type {
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

    /// 新しい盤面をデータベースに保存
    pub fn save_new_board(
        &self,
        board_sfen: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match &self.db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
                conn.execute(
                    "INSERT INTO training_data (board) VALUES (?1)",
                    [board_sfen],
                )?;
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
        Ok(())
    }

    /// 更新対象のレコードを読み取り
    pub fn read_records_for_update(
        &self,
        max_records: Option<usize>,
    ) -> Result<Vec<(i64, String)>, Box<dyn std::error::Error + Send + Sync>> {
        match &self.db_type {
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
                Ok(records)
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
                })
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }

    /// 対局結果をデータベースに書き込み
    pub fn update_game_results(
        &self,
        results: Vec<(i64, i32, i32, i32)>,
    ) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
        let mut updated_count = 0;
        match &self.db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
                for (id, white_wins, black_wins, total_games) in results {
                    let rows_affected = conn.execute(
                        "UPDATE training_data
                         SET white_wins = white_wins + ?1,
                             black_wins = black_wins + ?2,
                             total_games = total_games + ?3,
                             updated_at = CURRENT_TIMESTAMP
                         WHERE id = ?4",
                        [white_wins, black_wins, total_games, id as i32],
                    )?;
                    if rows_affected > 0 {
                        updated_count += 1;
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

                    for (id, white_wins, black_wins, total_games) in results {
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
                        if rows_affected > 0 {
                            updated_count += 1;
                        }
                    }
                    Ok::<(), tokio_postgres::Error>(())
                })?;
            }
        }
        Ok(updated_count)
    }

    /// 学習対象のレコード数を取得
    pub fn count_records_for_training(
        &self,
        min_games: i32,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        match &self.db_type {
            DatabaseType::Sqlite(db_path) => {
                let conn = rusqlite::Connection::open(db_path)?;
                let count: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM training_data WHERE total_games >= ?1",
                    [min_games],
                    |row| row.get(0),
                )?;
                Ok(count as usize)
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
                })
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    }
}
