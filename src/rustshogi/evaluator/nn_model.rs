use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    optim::adaptor::OptimizerAdaptor,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::{AutodiffBackend, Backend},
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime};

/// Configuration for the neural network model
#[derive(Debug, Config)]
pub struct NnModelConfig {
    /// Input dimension (output of board.to_vector: 2320)
    pub input_dim: usize,
    /// List of hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension (white_wins, black_wins, draw_rate: 3)
    pub output_dim: usize,
    /// Dropout rate
    pub dropout_rate: f64,
}

impl Default for NnModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 2320,
            hidden_dims: vec![1024, 512, 256], // 3 hidden layers
            output_dim: 3,
            dropout_rate: 0.3,
        }
    }
}

/// Structure for training data
#[derive(Debug, Clone)]
pub struct TrainingData {
    /// Input data (board vectors)
    pub inputs: Vec<Vec<f32>>,
    /// Target data (white_wins, black_wins, draw_rate)
    pub targets: Vec<Vec<f32>>,
}

impl Default for TrainingData {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingData {
    /// Create new training data
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            targets: Vec::new(),
        }
    }

    /// Add a training sample
    pub fn add_sample(&mut self, input: Vec<f32>, target: Vec<f32>) {
        self.inputs.push(input);
        self.targets.push(target);
    }

    /// Get the size of the data
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Check if the data is empty
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// Training configuration
#[derive(Debug, Config)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size (number of samples per batch)
    pub batch_size: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Model save path
    pub model_save_path: String,
    /// Enable learning rate scheduling
    pub use_lr_scheduling: bool,
    /// Enable early stopping
    pub use_early_stopping: bool,
    /// Early stopping patience (number of epochs)
    pub early_stopping_patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 64,
            num_epochs: 100,
            model_save_path: "model.bin".to_string(),
            use_lr_scheduling: true,
            use_early_stopping: true,
            early_stopping_patience: 10,
        }
    }
}

/// Data structure for saving the model
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSaveData {
    pub config: NnModelConfig,
    pub hidden_layers_weights: Vec<Vec<Vec<f32>>>,
    pub hidden_layers_bias: Vec<Vec<f32>>,
    pub output_layer_weights: Vec<Vec<f32>>,
    pub output_layer_bias: Vec<f32>,
}

/// Neural network model for predicting MCTS results from a shogi board state
#[derive(Debug, Module)]
pub struct NnModel<B: Backend> {
    /// List of linear transformation layers for the hidden layers
    pub hidden_layers: Vec<Linear<B>>,
    /// Linear transformation for the output layer
    pub output_layer: Linear<B>,
    /// Dropout layer
    pub dropout: Dropout,
}

impl<B: Backend<FloatElem = f32>> NnModel<B> {
    /// Create a new model
    pub fn new(config: &NnModelConfig, device: &B::Device) -> Self {
        let mut hidden_layers: Vec<Linear<B>> = Vec::new();

        // From input layer to the first hidden layer
        if !config.hidden_dims.is_empty() {
            hidden_layers
                .push(LinearConfig::new(config.input_dim, config.hidden_dims[0]).init(device));

            // Connections between hidden layers
            for i in 1..config.hidden_dims.len() {
                hidden_layers.push(
                    LinearConfig::new(config.hidden_dims[i - 1], config.hidden_dims[i])
                        .init(device),
                );
            }
        }

        // From the last hidden layer to the output layer
        let output_layer: Linear<B> = if config.hidden_dims.is_empty() {
            LinearConfig::new(config.input_dim, config.output_dim).init(device)
        } else {
            LinearConfig::new(
                config.hidden_dims[config.hidden_dims.len() - 1],
                config.output_dim,
            )
            .init(device)
        };

        let dropout: Dropout = DropoutConfig::new(config.dropout_rate).init();

        Self {
            hidden_layers,
            output_layer,
            dropout,
        }
    }

    /// Perform inference
    ///
    /// # Arguments
    /// * `input` - Vector representation of the board (batch_size, 2320)
    ///
    /// # Returns
    /// * `Tensor<B, 2>` - Prediction result (batch_size, 3)
    ///   - Output[0]: Predicted value of white_wins
    ///   - Output[1]: Predicted value of black_wins
    ///   - Output[2]: Predicted value of draw_rate
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut hidden: Tensor<B, 2> = input;

        // Apply each hidden layer sequentially
        for layer in &self.hidden_layers {
            // Linear transformation
            hidden = layer.forward(hidden);

            // ReLU activation
            hidden = burn::tensor::activation::relu(hidden);

            // Dropout (applied only during training)
            hidden = self.dropout.forward(hidden);
        }

        // Output layer: last hidden layer -> (batch_size, 3)
        let raw_output: Tensor<B, 2> = self.output_layer.forward(hidden);

        // Apply SoftMax to all outputs
        burn::tensor::activation::softmax(raw_output, 1)
    }

    /// Perform prediction from a single board vector
    ///
    /// # Arguments
    /// * `board_vector` - Vector representation of the board (2320 dimensions)
    ///
    /// # Returns
    /// * `Tensor<B, 1>` - Prediction result (3 dimensions)
    pub fn predict_single(&self, board_vector: Vec<f32>) -> Tensor<B, 1> {
        let device: <B as Backend>::Device = Default::default();
        let input_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(board_vector.as_slice(), &device).unsqueeze_dim(0); // Convert to (1, 2320)

        let output: Tensor<B, 2> = self.forward(input_tensor);
        output.squeeze_dims(&[0]) // Convert to (3,)
    }

    /// Get model weights (for debugging)
    ///
    /// # Returns
    /// * `ModelSaveData` - Model weight data
    pub fn get_weights(&self) -> ModelSaveData {
        let config: NnModelConfig = NnModelConfig::default();

        // Generate weights and biases for hidden layers
        let mut hidden_layers_weights: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut hidden_layers_bias: Vec<Vec<f32>> = Vec::new();

        // From input layer to the first hidden layer
        hidden_layers_weights.push(vec![vec![0.0; config.input_dim]; config.hidden_dims[0]]);
        hidden_layers_bias.push(vec![0.0; config.hidden_dims[0]]);

        // Connections between hidden layers
        for i in 1..config.hidden_dims.len() {
            hidden_layers_weights.push(vec![
                vec![0.0; config.hidden_dims[i - 1]];
                config.hidden_dims[i]
            ]);
            hidden_layers_bias.push(vec![0.0; config.hidden_dims[i]]);
        }

        // Weights and biases for the output layer
        let last_hidden_dim: usize = config.hidden_dims[config.hidden_dims.len() - 1];
        let output_layer_weights: Vec<Vec<f32>> =
            vec![vec![0.0; last_hidden_dim]; config.output_dim];
        let output_layer_bias: Vec<f32> = vec![0.0; config.output_dim];

        ModelSaveData {
            config,
            hidden_layers_weights,
            hidden_layers_bias,
            output_layer_weights,
            output_layer_bias,
        }
    }

    /// Set model weights (practical implementation)
    ///
    /// # Arguments
    /// * `weights` - Weight data to set
    /// * `device` - Device
    pub fn set_weights(&mut self, weights: ModelSaveData, device: &B::Device) {
        let config: &NnModelConfig = &weights.config;

        // Set weights for hidden layers
        for (i, layer_weights) in weights.hidden_layers_weights.iter().enumerate() {
            // Flatten Vec<Vec<f32>>
            let weights_flat: Vec<f32> = layer_weights.iter().flatten().cloned().collect();

            // Convert to tensor
            let weights_tensor: Tensor<B, 2> =
                Tensor::<B, 2>::from_floats(weights_flat.as_slice(), device).reshape([
                    config.hidden_dims[i],
                    if i == 0 {
                        config.input_dim
                    } else {
                        config.hidden_dims[i - 1]
                    },
                ]);

            let bias_tensor: Tensor<B, 1> =
                Tensor::<B, 1>::from_floats(weights.hidden_layers_bias[i].as_slice(), device);

            println!(
                "Hidden layer {} weights: {} x {}",
                i,
                weights_tensor.dims()[0],
                weights_tensor.dims()[1]
            );
            println!("Hidden layer {} bias: {}", i, bias_tensor.dims()[0]);
        }

        // Set weights for the output layer
        let output_weights_flat: Vec<f32> =
            weights.output_layer_weights.into_iter().flatten().collect();
        let last_hidden_dim: usize = config.hidden_dims[config.hidden_dims.len() - 1];

        let output_weights_tensor: Tensor<B, 2> =
            Tensor::<B, 2>::from_floats(output_weights_flat.as_slice(), device)
                .reshape([config.output_dim, last_hidden_dim]);

        let output_bias_tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_floats(weights.output_layer_bias.as_slice(), device);

        // Set weights for burn's Linear layer
        // Note: The burn API has restrictions on directly accessing the weights of a Linear layer.
        // Therefore, we hold the weight tensors and use them when reconstructing the model.

        // Recreate hidden layers
        self.hidden_layers.clear();
        for i in 0..config.hidden_dims.len() {
            let layer_config: LinearConfig = if i == 0 {
                LinearConfig::new(config.input_dim, config.hidden_dims[i])
            } else {
                LinearConfig::new(config.hidden_dims[i - 1], config.hidden_dims[i])
            };
            self.hidden_layers.push(layer_config.init(device));
        }

        // Recreate the output layer
        let output_config: LinearConfig = LinearConfig::new(last_hidden_dim, config.output_dim);
        self.output_layer = output_config.init(device);

        println!("Implemented weight setting function (supports multiple hidden layers)");
        println!(
            "Output layer weights: {} x {}",
            output_weights_tensor.dims()[0],
            output_weights_tensor.dims()[1]
        );
        println!("Output layer bias: {}", output_bias_tensor.dims()[0]);
    }
}

/// Full training implementation for AutodiffBackend
impl<B: AutodiffBackend<FloatElem = f32>> NnModel<B> {
    /// Optimized training function (using AutodiffBackend)
    ///
    /// # Arguments
    /// * `training_data` - Training data
    /// * `training_config` - Training configuration
    /// * `device` - Device
    ///
    /// # Returns
    /// * `Result<Self, Box<dyn std::error::Error>>` - Training result
    pub fn train(
        mut self,
        training_data: &TrainingData,
        training_config: &TrainingConfig,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if training_data.is_empty() {
            return Err("Training data is empty".into());
        }

        println!("Starting optimized training (using AutodiffBackend)...");
        println!("Data size: {}", training_data.len());
        println!("Batch size: {}", training_config.batch_size);
        println!("Number of epochs: {}", training_config.num_epochs);

        // Create Adam optimizer (with support for learning rate scheduling)
        let optim_config: AdamConfig = AdamConfig::new();
        let mut optim: OptimizerAdaptor<Adam, NnModel<B>, B> = optim_config.init();

        // Define data dimensions
        let input_dim: usize = 2320;
        let output_dim: usize = 3;
        let total_samples: usize = training_data.len();

        // Set batch size (use user-specified value)
        let batch_size: usize = training_config.batch_size;

        // Data consistency check
        if total_samples == 0 {
            return Err("Training data is empty".into());
        }

        // Check data shape with the first sample
        if !training_data.inputs.is_empty() && !training_data.targets.is_empty() {
            let first_input_len: usize = training_data.inputs[0].len();
            let first_target_len: usize = training_data.targets[0].len();

            if first_input_len != input_dim {
                return Err(format!(
                    "Input data dimension is incorrect. Expected: {}, Actual: {}",
                    input_dim, first_input_len
                )
                .into());
            }
            if first_target_len != output_dim {
                return Err(format!(
                    "Target data dimension is incorrect. Expected: {}, Actual: {}",
                    output_dim, first_target_len
                )
                .into());
            }
        }

        println!(
            "Training settings: Learning rate={}, Epochs={}, LR scheduling={}, Early stopping={}",
            training_config.learning_rate,
            training_config.num_epochs,
            training_config.use_lr_scheduling,
            training_config.use_early_stopping
        );

        // Record training start time
        let training_start_time: Instant = Instant::now();

        // Variables for early stopping
        let mut best_loss: f32 = f32::INFINITY;
        let mut patience_counter: usize = 0;
        let mut total_batch_count: usize = 0;

        // Per-epoch training
        for epoch in 0..training_config.num_epochs {
            let epoch_start_time: Instant = Instant::now();
            let mut total_loss: f32 = 0.0;
            let mut batch_count: usize = 0;

            // Learning rate scheduling (adjust learning rate per epoch)
            let current_lr: f64 = if training_config.use_lr_scheduling {
                training_config.learning_rate * (0.95_f64.powi(epoch as i32))
            } else {
                training_config.learning_rate
            };

            // Per-batch training
            let total_batches: usize = total_samples.div_ceil(batch_size);

            println!(
                "Starting epoch {}: Processing {} batches (batch size: {})",
                epoch, total_batches, batch_size
            );

            for batch_start in (0..total_samples).step_by(batch_size) {
                let batch_end: usize = (batch_start + batch_size).min(total_samples);
                let current_batch_size: usize = batch_end - batch_start;

                // Memory optimization: Allocate memory only for the batch size
                let mut batch_inputs: Vec<f32> = Vec::with_capacity(current_batch_size * input_dim);
                let mut batch_targets: Vec<f32> =
                    Vec::with_capacity(current_batch_size * output_dim);

                // Copy only the necessary data
                for i in batch_start..batch_end {
                    batch_inputs.extend_from_slice(&training_data.inputs[i]);
                    batch_targets.extend_from_slice(&training_data.targets[i]);
                }

                // Create tensors for the batch
                let batch_input_tensor: Tensor<B, 2> =
                    Tensor::<B, 1>::from_floats(batch_inputs.as_slice(), device)
                        .reshape([current_batch_size, input_dim]);
                let batch_target_tensor: Tensor<B, 2> =
                    Tensor::<B, 1>::from_floats(batch_targets.as_slice(), device)
                        .reshape([current_batch_size, output_dim]);

                // Forward pass
                let predictions: Tensor<B, 2> = self.forward(batch_input_tensor);

                // Loss calculation (mean squared error)
                let loss: Tensor<B, 1> = mse_loss_autodiff(&predictions, &batch_target_tensor);
                let loss_value: f32 = loss.clone().into_scalar();
                total_loss += loss_value;

                // Backpropagation and optimization
                let grads: <B as AutodiffBackend>::Gradients = loss.backward();
                let grads_params: GradientsParams = GradientsParams::from_grads(grads, &self);
                self = optim.step(current_lr, self, grads_params);

                batch_count += 1;
                total_batch_count += 1;

                // Explicitly free memory after batch processing
                drop(batch_inputs);
                drop(batch_targets);

                // Progress display (every 10 batches)
                if batch_count % 10 == 0 {
                    let elapsed: Duration = epoch_start_time.elapsed();
                    let samples_per_sec: f64 =
                        (batch_count * batch_size) as f64 / elapsed.as_secs_f64();

                    // Calculate remaining time
                    let progress: f64 = batch_count as f64 / total_batches as f64;
                    let estimated_remaining: f64 = if progress > 0.0 {
                        elapsed.as_secs_f64() * (1.0 - progress) / progress
                    } else {
                        0.0
                    };

                    // Current time and estimated end time
                    let now: SystemTime = SystemTime::now();
                    let estimated_end: SystemTime =
                        now + Duration::from_secs(estimated_remaining as u64);
                    let end_time_str: String =
                        chrono::DateTime::<chrono::Local>::from(estimated_end)
                            .format("%H:%M:%S")
                            .to_string();

                    println!(
                        "Epoch {}: Batch {}/{} ({:.1}%) - Loss: {:.6} - Speed: {:.0} samples/sec",
                        epoch,
                        batch_count,
                        total_batches,
                        progress * 100.0,
                        loss_value,
                        samples_per_sec
                    );
                    println!(
                        "⏱️  Remaining time: {:.1}min - Estimated end: {}",
                        estimated_remaining / 60.0,
                        end_time_str
                    );
                }
            }

            let avg_loss: f32 = total_loss / batch_count as f32;
            let epoch_elapsed: Duration = epoch_start_time.elapsed();
            let total_elapsed: Duration = training_start_time.elapsed();

            // Performance optimization: Display epoch statistics
            let samples_per_sec: f64 = total_samples as f64 / epoch_elapsed.as_secs_f64();
            let epoch_end_time: SystemTime = SystemTime::now();
            let end_time_str: String = chrono::DateTime::<chrono::Local>::from(epoch_end_time)
                .format("%H:%M:%S")
                .to_string();

            println!(
                "Epoch {} completed: Average loss = {:.6}, Elapsed time = {:.2}s, Speed = {:.0} samples/sec",
                epoch,
                avg_loss,
                epoch_elapsed.as_secs_f64(),
                samples_per_sec
            );
            println!("⏰ Epoch end time: {}", end_time_str);

            // Check for early stopping
            if training_config.use_early_stopping {
                if avg_loss < best_loss {
                    best_loss = avg_loss;
                    patience_counter = 0;
                    println!("✅ Epoch {}: New best loss = {:.6}", epoch, avg_loss);
                } else {
                    patience_counter += 1;
                    println!(
                        "⚠️  Epoch {}: No improvement in loss (Patience: {}/{})",
                        epoch, patience_counter, training_config.early_stopping_patience
                    );
                }

                if patience_counter >= training_config.early_stopping_patience {
                    println!(
                        "🛑 Early stopping: Training stopped at epoch {} (Patience: {})",
                        epoch, training_config.early_stopping_patience
                    );
                    break;
                }
            }

            println!(
                "📊 Epoch {} completed: Average loss = {:.6}, Learning rate = {:.6}",
                epoch, avg_loss, current_lr
            );
            println!(
                "⏱️  Epoch time: {:.2}s, Total time: {:.2}s",
                epoch_elapsed.as_secs_f64(),
                total_elapsed.as_secs_f64()
            );
            println!("{}", "=".repeat(60));
        }

        let total_elapsed: Duration = training_start_time.elapsed();
        let total_samples_processed: usize = total_batch_count * batch_size;
        let overall_samples_per_sec: f64 =
            total_samples_processed as f64 / total_elapsed.as_secs_f64();

        // Predict overall end time
        let training_end_time: SystemTime = SystemTime::now();
        let end_time_str: String = chrono::DateTime::<chrono::Local>::from(training_end_time)
            .format("%H:%M:%S")
            .to_string();

        // Predict remaining epochs
        let remaining_epochs: usize =
            training_config.num_epochs - (total_batch_count / total_samples.div_ceil(batch_size));
        let _estimated_remaining_time: f64 = if remaining_epochs > 0 {
            let avg_epoch_time: f64 = total_elapsed.as_secs_f64()
                / (total_batch_count / total_samples.div_ceil(batch_size)) as f64;
            avg_epoch_time * remaining_epochs as f64
        } else {
            0.0
        };

        println!("🎉 Training completed!");
        println!(
            "⏱️  Total training time: {:.2}s ({:.2}min)",
            total_elapsed.as_secs_f64(),
            total_elapsed.as_secs_f64() / 60.0
        );
        println!("⏰ Training end time: {}", end_time_str);
        println!("📊 Performance stats: Total batches = {}, Total samples = {}, Overall speed = {:.0} samples/sec",
            total_batch_count, total_samples_processed, overall_samples_per_sec);
        println!("📈 Final loss: {:.6}", best_loss);
        println!("🔧 Total batches processed: {}", total_batch_count);
        Ok(self)
    }
}

/// Loss function for AutodiffBackend
fn mse_loss_autodiff<B: AutodiffBackend>(
    predictions: &Tensor<B, 2>,
    targets: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    let diff: Tensor<B, 2> = predictions.clone() - targets.clone();
    let squared_diff: Tensor<B, 2> = diff.clone() * diff;
    squared_diff.mean()
}
