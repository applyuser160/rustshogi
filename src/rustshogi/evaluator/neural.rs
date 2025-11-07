use super::super::board::Board;
use super::super::color::ColorType;
use super::super::game::Game;
use super::abst::Evaluator;
use super::database::{DatabaseType, TrainingDatabase, TrainingRecord};
use super::nn_model::{NnModel, NnModelConfig, TrainingConfig};
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
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

/// Neural network evaluation function system
#[pyclass]
#[derive(Clone)]
pub struct NeuralEvaluator {
    db_type: Option<DatabaseType>,
    model_path: Option<String>,
}

impl NeuralEvaluator {
    /// Create a new evaluation function system
    pub fn new(db_type: Option<DatabaseType>, model_path: Option<String>) -> Self {
        Self {
            db_type,
            model_path,
        }
    }

    /// Helper for database operations
    fn get_db(&self) -> Result<TrainingDatabase, Box<dyn std::error::Error + Send + Sync>> {
        let db_type = self
            .db_type
            .as_ref()
            .cloned()
            .ok_or("Database type is not set")?;
        Ok(TrainingDatabase::new(db_type))
    }

    /// Initialize the database table
    pub fn init_database(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.get_db()?.init_database()
    }

    /// Generate and save random boards to the RDB
    pub fn generate_and_save_random_boards(
        &self,
        count: usize,
    ) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
        let db = self.get_db()?;
        let mut saved_count = 0;
        let start_time = Instant::now();

        println!("Generating {} random boards...", count);

        for i in 0..count {
            let mut game = Game::new();
            game.input_board("startpos".to_string());
            let random_board = game.generate_random_board();
            let board_sfen = random_board.to_string();

            db.save_new_board(&board_sfen)?;
            saved_count += 1;

            if (i + 1) % 100 == 0 {
                let elapsed = start_time.elapsed();
                println!(
                    "Generated and saved {} boards (elapsed time: {:.2}s)",
                    i + 1,
                    elapsed.as_secs_f64()
                );
            }
        }

        let total_elapsed = start_time.elapsed();
        println!(
            "Generated and saved {} random boards (total time: {:.2}s)",
            saved_count,
            total_elapsed.as_secs_f64()
        );
        Ok(saved_count)
    }

    /// Read saved records, execute random games, and update the win counts
    pub fn update_records_with_random_games(
        &self,
        trials_per_record: usize,
        max_records: Option<usize>,
        num_threads: usize,
    ) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
        let db = self.get_db()?;
        let start_time = Instant::now();
        let records = db.read_records_for_update(max_records)?;

        println!(
            "Processing {} records sequentially (with {} threads in parallel for each record)",
            records.len(),
            num_threads
        );

        match self.get_database_stats() {
            Ok((total_records, total_games, avg_games)) => {
                println!(
                    "Database statistics before processing: Total records={}, Total games={}, Average games={}",
                    total_records, total_games, avg_games
                );
            }
            Err(e) => {
                eprintln!("Failed to get database statistics: {}", e);
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
                    "Progress: {}/{} ({:.1}%) - Elapsed time: {:.2}s",
                    processed_count,
                    total_records,
                    progress_percent,
                    elapsed.as_secs_f64()
                );
            }
        }

        println!(
            "Random game trials completed. Processed records: {}",
            all_results.len()
        );

        println!("Starting to write to the database...");
        let updated_count = db.update_game_results(all_results)?;
        println!(
            "Finished writing to the database. Updated records: {}",
            updated_count
        );

        match self.get_database_stats() {
            Ok((total_records, total_games, avg_games)) => {
                println!(
                    "Database statistics: Total records={}, Total games={}, Average games={}",
                    total_records, total_games, avg_games
                );
            }
            Err(e) => {
                eprintln!("Failed to get database statistics: {}", e);
            }
        }

        let total_elapsed = start_time.elapsed();
        println!(
            "Updated {} records (total time: {:.2}s)",
            updated_count,
            total_elapsed.as_secs_f64()
        );
        Ok(updated_count)
    }

    /// Loss function (for AutodiffBackend) - Cross Entropy Loss
    /// Suitable for predicting probability distributions (white_win_rate, black_win_rate, draw_rate)
    /// Treats each element as an independent probability and calculates binary cross entropy
    fn cross_entropy_loss_autodiff<B: AutodiffBackend>(
        predictions: &Tensor<B, 2>,
        targets: &Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let epsilon = 1e-5;
        // Clip predictions to [epsilon, 1.0 - epsilon] range for numerical stability
        let clipped_pred = predictions.clone().clamp(epsilon, 1.0 - epsilon);
        // Calculate log of predictions
        let log_pred = clipped_pred.clone().log();
        // Calculate log of (1 - pred) using tensor subtraction
        let ones_pred = Tensor::ones_like(&clipped_pred);
        let one_minus_pred = ones_pred - clipped_pred;
        let log_one_minus_pred = one_minus_pred.log();
        // Binary cross entropy: -(target * log(pred) + (1 - target) * log(1 - pred))
        let ones_target = Tensor::ones_like(targets);
        let neg_log_likelihood =
            targets.clone() * log_pred + (ones_target - targets.clone()) * log_one_minus_pred;
        // Calculate average over the entire batch
        neg_log_likelihood.sum().neg() / (predictions.dims()[0] as f32)
    }

    /// MSE Loss function (for backward compatibility)
    #[allow(dead_code)]
    fn mse_loss_autodiff<B: AutodiffBackend>(
        predictions: &Tensor<B, 2>,
        targets: &Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let diff = predictions.clone() - targets.clone();
        let squared_diff = diff.clone() * diff;
        squared_diff.mean()
    }

    /// Model initialization process (load existing model or create a new one)
    fn initialize_model(
        model_save_path: &str,
        device: &NdArrayDevice,
    ) -> Result<NnModel<Autodiff<NdArray>>, Box<dyn std::error::Error + Send + Sync>> {
        let model_config = NnModelConfig::default();
        let model = if Path::new(model_save_path).exists() {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            NnModel::new(&model_config, device).load_file(model_save_path, &recorder, device)?
        } else {
            NnModel::new(&model_config, device)
        };
        Ok(model)
    }

    /// Process one batch of training
    fn process_training_batch(
        model: NnModel<Autodiff<NdArray>>,
        batch_inputs: &[Vec<f32>],
        batch_targets: &[Vec<f32>],
        device: &NdArrayDevice,
        optim: &mut impl Optimizer<NnModel<Autodiff<NdArray>>, Autodiff<NdArray>>,
        current_lr: f64,
    ) -> (NnModel<Autodiff<NdArray>>, f32) {
        let (input_dim, output_dim) = (2320, 3);
        let batch_size = batch_inputs.len();

        let mut flat_inputs = Vec::with_capacity(batch_size * input_dim);
        let mut flat_targets = Vec::with_capacity(batch_size * output_dim);

        for i in 0..batch_size {
            flat_inputs.extend_from_slice(&batch_inputs[i]);
            flat_targets.extend_from_slice(&batch_targets[i]);
        }

        let batch_input_tensor =
            Tensor::<Autodiff<NdArray>, 1>::from_floats(flat_inputs.as_slice(), device)
                .reshape([batch_size, input_dim]);

        let batch_target_tensor =
            Tensor::<Autodiff<NdArray>, 1>::from_floats(flat_targets.as_slice(), device)
                .reshape([batch_size, output_dim]);

        let predictions = model.forward(batch_input_tensor);
        let loss = Self::cross_entropy_loss_autodiff(&predictions, &batch_target_tensor);
        let loss_value: f32 = loss.clone().into_scalar();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        let updated_model = optim.step(current_lr, model, grads_params);

        (updated_model, loss_value)
    }

    /// Update progress display
    fn update_progress_display(
        processed_samples: usize,
        target_count: usize,
        epoch_start_time: &Instant,
    ) {
        let progress_percent = (processed_samples as f64 / target_count as f64) * 100.0;
        let elapsed = epoch_start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

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

        let now = std::time::SystemTime::now();
        let estimated_end = now + std::time::Duration::from_secs(estimated_remaining_secs as u64);
        let end_time_str = chrono::DateTime::<chrono::Local>::from(estimated_end)
            .format("%H:%M:%S")
            .to_string();

        print!(
            "\r Progress: {}/{} ({:.1}%) - Elapsed: {:.1}s - Speed: {:.0} samples/sec - Remaining: {:.1}min (ETA: {})",
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

    /// Save model checkpoint
    fn save_model_checkpoint(
        model: &NnModel<Autodiff<NdArray>>,
        model_save_path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(model_save_path, &recorder)?;
        println!("Model saved to: {}", model_save_path);
        Ok(())
    }

    /// Get training data and train the model (using all data with streaming)
    pub fn train_model(
        &self,
        min_games: i32,
        training_config: TrainingConfig,
        model_save_path: String,
        max_samples: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = self.get_db()?;
        let start_time = Instant::now();
        println!("Starting model training...");

        let total_count = db.count_records_for_training(min_games)?;
        let target_count = max_samples.unwrap_or(total_count).min(total_count);
        println!("Total records: {} (using: {})", total_count, target_count);

        if target_count == 0 {
            return Err("No training data found".into());
        }

        const DB_FETCH_BATCH_SIZE: usize = 1000;
        let device = NdArrayDevice::default();
        let mut model = Self::initialize_model(&model_save_path, &device)?;

        let optim_config = AdamConfig::new();
        let mut optim = optim_config.init();

        println!(
            "Starting training: Epochs={}, Batch size={}",
            training_config.num_epochs, training_config.batch_size
        );

        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..training_config.num_epochs {
            let epoch_start_time = Instant::now();
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            let current_lr = if training_config.use_lr_scheduling {
                training_config.learning_rate * (0.95_f64.powi(epoch as i32))
            } else {
                training_config.learning_rate
            };

            println!("Starting epoch {}...", epoch);

            let mut db_offset = 0;
            let mut processed_samples = 0;
            const PROGRESS_UPDATE_INTERVAL: usize = 1000;

            loop {
                let mut batch_records =
                    db.fetch_batch_from_db(min_games, DB_FETCH_BATCH_SIZE, db_offset)?;
                if batch_records.is_empty() {
                    break;
                }

                // Shuffle data to randomize training order
                let mut rng = thread_rng();
                batch_records.shuffle(&mut rng);

                // Parallel preprocessing
                let results: Vec<_> = batch_records
                    .par_iter()
                    .map(|(board_sfen, white_wins, black_wins, total_games)| {
                        let board = Board::from_sfen(board_sfen.clone());
                        let input = board.to_vector(None);

                        let total_games_f = *total_games as f32;
                        let white_win_rate = if *total_games > 0 {
                            *white_wins as f32 / total_games_f
                        } else {
                            0.5
                        };
                        let black_win_rate = if *total_games > 0 {
                            *black_wins as f32 / total_games_f
                        } else {
                            0.5
                        };
                        let draw_rate = if *total_games > 0 {
                            (total_games_f - *white_wins as f32 - *black_wins as f32)
                                / total_games_f
                        } else {
                            0.0
                        };

                        (input, vec![white_win_rate, black_win_rate, draw_rate])
                    })
                    .collect();

                let mut batch_inputs = Vec::with_capacity(results.len());
                let mut batch_targets = Vec::with_capacity(results.len());

                for (input, target) in results {
                    batch_inputs.push(input);
                    batch_targets.push(target);
                }

                for batch_start in (0..batch_inputs.len()).step_by(training_config.batch_size) {
                    let batch_end =
                        (batch_start + training_config.batch_size).min(batch_inputs.len());
                    let current_batch_size = batch_end - batch_start;

                    let (updated_model, loss_value) = Self::process_training_batch(
                        model,
                        &batch_inputs[batch_start..batch_end],
                        &batch_targets[batch_start..batch_end],
                        &device,
                        &mut optim,
                        current_lr,
                    );
                    model = updated_model;
                    total_loss += loss_value;

                    batch_count += 1;
                    processed_samples += current_batch_size;

                    if processed_samples % PROGRESS_UPDATE_INTERVAL == 0
                        || processed_samples >= target_count
                    {
                        Self::update_progress_display(
                            processed_samples,
                            target_count,
                            &epoch_start_time,
                        );
                    }
                }

                db_offset += DB_FETCH_BATCH_SIZE;
                if db_offset >= target_count {
                    break;
                }
            }
            println!();

            let avg_loss = total_loss / batch_count as f32;
            println!(
                "Epoch {} completed: Average loss = {:.6}, Elapsed time = {:.2}s",
                epoch,
                avg_loss,
                epoch_start_time.elapsed().as_secs_f64()
            );

            if training_config.use_early_stopping {
                if avg_loss < best_loss {
                    best_loss = avg_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= training_config.early_stopping_patience {
                        println!("Early stopping: Stopping training at epoch {}", epoch);
                        break;
                    }
                }
            }

            if let Err(e) = Self::save_model_checkpoint(&model, &model_save_path) {
                eprintln!("Failed to save model: {}", e);
            }
        }

        println!("Training completed");
        if let Err(e) = Self::save_model_checkpoint(&model, &model_save_path) {
            eprintln!("Failed to save final model: {}", e);
        }

        println!(
            "Total model training time: {:.2}s",
            start_time.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Load the model and perform inference on any board state (evaluation function execution)
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
                .ok_or("Model path is not set")?
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

    /// Get detailed information for a specific record
    pub fn get_record_details(
        &self,
        record_id: i64,
    ) -> Result<Option<TrainingRecord>, Box<dyn std::error::Error + Send + Sync>> {
        self.get_db()?.get_record_details(record_id)
    }

    /// Get database statistics
    pub fn get_database_stats(
        &self,
    ) -> Result<(i64, i64, i64), Box<dyn std::error::Error + Send + Sync>> {
        self.get_db()?.get_database_stats()
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
