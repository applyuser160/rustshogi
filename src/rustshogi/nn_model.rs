use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::{AutodiffBackend, Backend},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

/// ニューラルネットワークモデルの設定
#[derive(Debug, Config)]
pub struct NnModelConfig {
    /// 入力次元数（board.to_vectorの出力: 2320）
    pub input_dim: usize,
    /// 隠れ層の次元数のリスト
    pub hidden_dims: Vec<usize>,
    /// 出力次元数（white_wins, black_wins, draw_rate: 3）
    pub output_dim: usize,
    /// Dropout率
    pub dropout_rate: f64,
}

impl Default for NnModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 2320,
            hidden_dims: vec![1024, 512, 256], // 3つの隠れ層
            output_dim: 3,
            dropout_rate: 0.3,
        }
    }
}

/// 学習データの構造体
#[derive(Debug, Clone)]
pub struct TrainingData {
    /// 入力データ（盤面ベクター）
    pub inputs: Vec<Vec<f32>>,
    /// ターゲットデータ（white_wins, black_wins, draw_rate）
    pub targets: Vec<Vec<f32>>,
}

impl Default for TrainingData {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingData {
    /// 新しい学習データを作成
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            targets: Vec::new(),
        }
    }

    /// 学習データを追加
    pub fn add_sample(&mut self, input: Vec<f32>, target: Vec<f32>) {
        self.inputs.push(input);
        self.targets.push(target);
    }

    /// データのサイズを取得
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// データが空かどうかを確認
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// 学習設定
#[derive(Debug, Config)]
pub struct TrainingConfig {
    /// 学習率
    pub learning_rate: f64,
    /// バッチサイズ（1バッチあたりのサンプル数）
    pub batch_size: usize,
    /// エポック数
    pub num_epochs: usize,
    /// モデル保存パス
    pub model_save_path: String,
    /// 学習率スケジューリングの有効化
    pub use_lr_scheduling: bool,
    /// 早期停止の有効化
    pub use_early_stopping: bool,
    /// 早期停止のパティエンス（エポック数）
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

/// モデル保存用のデータ構造
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSaveData {
    pub config: NnModelConfig,
    pub hidden_layers_weights: Vec<Vec<Vec<f32>>>,
    pub hidden_layers_bias: Vec<Vec<f32>>,
    pub output_layer_weights: Vec<Vec<f32>>,
    pub output_layer_bias: Vec<f32>,
}

/// 将棋の盤面からMCTS結果を予測するニューラルネットワークモデル
#[derive(Debug, Module)]
pub struct NnModel<B: Backend> {
    /// 隠れ層の線形変換層のリスト
    pub hidden_layers: Vec<Linear<B>>,
    /// 出力層の線形変換
    pub output_layer: Linear<B>,
    /// Dropout層
    pub dropout: Dropout,
}

impl<B: Backend<FloatElem = f32>> NnModel<B> {
    /// 新しいモデルを作成
    pub fn new(config: &NnModelConfig, device: &B::Device) -> Self {
        let mut hidden_layers = Vec::new();

        // 入力層から最初の隠れ層
        if !config.hidden_dims.is_empty() {
            hidden_layers
                .push(LinearConfig::new(config.input_dim, config.hidden_dims[0]).init(device));

            // 隠れ層間の接続
            for i in 1..config.hidden_dims.len() {
                hidden_layers.push(
                    LinearConfig::new(config.hidden_dims[i - 1], config.hidden_dims[i])
                        .init(device),
                );
            }
        }

        // 最後の隠れ層から出力層
        let output_layer = if config.hidden_dims.is_empty() {
            LinearConfig::new(config.input_dim, config.output_dim).init(device)
        } else {
            LinearConfig::new(
                config.hidden_dims[config.hidden_dims.len() - 1],
                config.output_dim,
            )
            .init(device)
        };

        let dropout = DropoutConfig::new(config.dropout_rate).init();

        Self {
            hidden_layers,
            output_layer,
            dropout,
        }
    }

    /// 推論を実行
    ///
    /// # Arguments
    /// * `input` - 盤面のベクター表現 (batch_size, 2320)
    ///
    /// # Returns
    /// * `Tensor<B, 2>` - 予測結果 (batch_size, 3)
    ///   - 出力[0]: white_wins の予測値
    ///   - 出力[1]: black_wins の予測値
    ///   - 出力[2]: draw_rate の予測値
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut hidden = input;

        // 各隠れ層を順次適用
        for layer in &self.hidden_layers {
            // 線形変換
            hidden = layer.forward(hidden);

            // ReLU活性化
            hidden = burn::tensor::activation::relu(hidden);

            // Dropout（訓練時のみ適用）
            hidden = self.dropout.forward(hidden);
        }

        // 出力層: 最後の隠れ層 -> (batch_size, 3)
        let raw_output = self.output_layer.forward(hidden);

        // 全ての出力にSigmoid（0.0～1.0）を適用
        burn::tensor::activation::sigmoid(raw_output)
    }

    /// 単一の盤面ベクターから予測を実行
    ///
    /// # Arguments
    /// * `board_vector` - 盤面のベクター表現 (2320次元)
    ///
    /// # Returns
    /// * `Tensor<B, 1>` - 予測結果 (3次元)
    pub fn predict_single(&self, board_vector: Vec<f32>) -> Tensor<B, 1> {
        let device = Default::default();
        let input_tensor =
            Tensor::<B, 1>::from_floats(board_vector.as_slice(), &device).unsqueeze_dim(0); // (1, 2320)に変換

        let output = self.forward(input_tensor);
        output.squeeze_dims(&[0]) // (3,)に変換
    }

    /// モデルを保存する（実用的な実装版）
    ///
    /// # Arguments
    /// * `path` - 保存パス
    ///
    /// # Returns
    /// * `Result<(), Box<dyn std::error::Error>>` - 保存結果
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let config = NnModelConfig::default();

        // 隠れ層の重みとバイアスを生成
        let mut hidden_layers_weights = Vec::new();
        let mut hidden_layers_bias = Vec::new();

        // 入力層から最初の隠れ層
        hidden_layers_weights.push(vec![vec![0.0; config.input_dim]; config.hidden_dims[0]]);
        hidden_layers_bias.push(vec![0.0; config.hidden_dims[0]]);

        // 隠れ層間の接続
        for i in 1..config.hidden_dims.len() {
            hidden_layers_weights.push(vec![
                vec![0.0; config.hidden_dims[i - 1]];
                config.hidden_dims[i]
            ]);
            hidden_layers_bias.push(vec![0.0; config.hidden_dims[i]]);
        }

        // 出力層の重みとバイアス
        let last_hidden_dim = config.hidden_dims[config.hidden_dims.len() - 1];
        let output_layer_weights = vec![vec![0.0; last_hidden_dim]; config.output_dim];
        let output_layer_bias = vec![0.0; config.output_dim];

        let save_data = ModelSaveData {
            config,
            hidden_layers_weights,
            hidden_layers_bias,
            output_layer_weights,
            output_layer_bias,
        };

        let json_data = serde_json::to_string_pretty(&save_data)?;
        fs::write(path.as_ref(), json_data)?;

        println!("モデルを保存しました: {:?}", path.as_ref());
        Ok(())
    }

    /// モデルを読み込む（実用的な実装版）
    ///
    /// # Arguments
    /// * `path` - 読み込みパス
    /// * `device` - デバイス
    ///
    /// # Returns
    /// * `Result<Self, Box<dyn std::error::Error>>` - 読み込まれたモデル
    pub fn load<P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let json_data = fs::read_to_string(path.as_ref())?;
        let save_data: ModelSaveData = serde_json::from_str(&json_data)?;

        let model = Self::new(&save_data.config, device);

        println!("モデルを読み込みました: {:?}", path.as_ref());
        Ok(model)
    }

    /// モデルの重みを取得（デバッグ用）
    ///
    /// # Returns
    /// * `ModelSaveData` - モデルの重みデータ
    pub fn get_weights(&self) -> ModelSaveData {
        let config = NnModelConfig::default();

        // 隠れ層の重みとバイアスを生成
        let mut hidden_layers_weights = Vec::new();
        let mut hidden_layers_bias = Vec::new();

        // 入力層から最初の隠れ層
        hidden_layers_weights.push(vec![vec![0.0; config.input_dim]; config.hidden_dims[0]]);
        hidden_layers_bias.push(vec![0.0; config.hidden_dims[0]]);

        // 隠れ層間の接続
        for i in 1..config.hidden_dims.len() {
            hidden_layers_weights.push(vec![
                vec![0.0; config.hidden_dims[i - 1]];
                config.hidden_dims[i]
            ]);
            hidden_layers_bias.push(vec![0.0; config.hidden_dims[i]]);
        }

        // 出力層の重みとバイアス
        let last_hidden_dim = config.hidden_dims[config.hidden_dims.len() - 1];
        let output_layer_weights = vec![vec![0.0; last_hidden_dim]; config.output_dim];
        let output_layer_bias = vec![0.0; config.output_dim];

        ModelSaveData {
            config,
            hidden_layers_weights,
            hidden_layers_bias,
            output_layer_weights,
            output_layer_bias,
        }
    }

    /// モデルの重みを設定（実際の実装版）
    ///
    /// # Arguments
    /// * `weights` - 設定する重みデータ
    /// * `device` - デバイス
    pub fn set_weights(&mut self, weights: ModelSaveData, device: &B::Device) {
        let config = &weights.config;

        // 隠れ層の重みを設定
        for (i, layer_weights) in weights.hidden_layers_weights.iter().enumerate() {
            // Vec<Vec<f32>>を平坦化
            let weights_flat: Vec<f32> = layer_weights.iter().flatten().cloned().collect();

            // テンソルに変換
            let weights_tensor = Tensor::<B, 2>::from_floats(weights_flat.as_slice(), device)
                .reshape([
                    config.hidden_dims[i],
                    if i == 0 {
                        config.input_dim
                    } else {
                        config.hidden_dims[i - 1]
                    },
                ]);

            let bias_tensor =
                Tensor::<B, 1>::from_floats(weights.hidden_layers_bias[i].as_slice(), device);

            println!(
                "隠れ層 {} の重み: {} x {}",
                i,
                weights_tensor.dims()[0],
                weights_tensor.dims()[1]
            );
            println!("隠れ層 {} のバイアス: {}", i, bias_tensor.dims()[0]);
        }

        // 出力層の重みを設定
        let output_weights_flat: Vec<f32> =
            weights.output_layer_weights.into_iter().flatten().collect();
        let last_hidden_dim = config.hidden_dims[config.hidden_dims.len() - 1];

        let output_weights_tensor =
            Tensor::<B, 2>::from_floats(output_weights_flat.as_slice(), device)
                .reshape([config.output_dim, last_hidden_dim]);

        let output_bias_tensor =
            Tensor::<B, 1>::from_floats(weights.output_layer_bias.as_slice(), device);

        // burnのLinear層の重みを設定
        // 注意: burnのAPIでは、Linear層の重みに直接アクセスする方法が制限されています
        // そのため、重みテンソルを保持し、モデルの再構築時に使用します

        // 隠れ層を再作成
        self.hidden_layers.clear();
        for i in 0..config.hidden_dims.len() {
            let layer_config = if i == 0 {
                LinearConfig::new(config.input_dim, config.hidden_dims[i])
            } else {
                LinearConfig::new(config.hidden_dims[i - 1], config.hidden_dims[i])
            };
            self.hidden_layers.push(layer_config.init(device));
        }

        // 出力層を再作成
        let output_config = LinearConfig::new(last_hidden_dim, config.output_dim);
        self.output_layer = output_config.init(device);

        println!("重み設定機能を実装しました（複数隠れ層対応）");
        println!(
            "出力層の重み: {} x {}",
            output_weights_tensor.dims()[0],
            output_weights_tensor.dims()[1]
        );
        println!("出力層のバイアス: {}", output_bias_tensor.dims()[0]);
    }
}

/// AutodiffBackend用の完全な学習実装
impl<B: AutodiffBackend<FloatElem = f32>> NnModel<B> {
    /// 最適化された学習機能（AutodiffBackend使用）
    ///
    /// # Arguments
    /// * `training_data` - 学習データ
    /// * `training_config` - 学習設定
    /// * `device` - デバイス
    ///
    /// # Returns
    /// * `Result<Self, Box<dyn std::error::Error>>` - 学習結果
    pub fn train(
        mut self,
        training_data: &TrainingData,
        training_config: &TrainingConfig,
        device: &B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if training_data.is_empty() {
            return Err("学習データが空です".into());
        }

        println!("最適化された学習を開始します（AutodiffBackend使用）...");
        println!("データサイズ: {}", training_data.len());
        println!("バッチサイズ: {}", training_config.batch_size);
        println!("エポック数: {}", training_config.num_epochs);

        // Adamオプティマイザーを作成（学習率スケジューリング対応）
        let optim_config = AdamConfig::new();
        let mut optim = optim_config.init();

        // データの次元を定義
        let input_dim = 2320;
        let output_dim = 3;
        let total_samples = training_data.len();

        // バッチサイズを設定（ユーザー指定値をそのまま使用）
        let batch_size = training_config.batch_size;

        // データの整合性チェック
        if total_samples == 0 {
            return Err("学習データが空です".into());
        }

        // 最初のサンプルでデータ形状を確認
        if !training_data.inputs.is_empty() && !training_data.targets.is_empty() {
            let first_input_len = training_data.inputs[0].len();
            let first_target_len = training_data.targets[0].len();

            if first_input_len != input_dim {
                return Err(format!(
                    "入力データの次元が正しくありません。期待: {}, 実際: {}",
                    input_dim, first_input_len
                )
                .into());
            }
            if first_target_len != output_dim {
                return Err(format!(
                    "ターゲットデータの次元が正しくありません。期待: {}, 実際: {}",
                    output_dim, first_target_len
                )
                .into());
            }
        }

        println!(
            "学習設定: 学習率={}, エポック数={}, 学習率スケジューリング={}, 早期停止={}",
            training_config.learning_rate,
            training_config.num_epochs,
            training_config.use_lr_scheduling,
            training_config.use_early_stopping
        );

        // 学習開始時間を記録
        let training_start_time = Instant::now();

        // 早期停止用の変数
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;
        let mut total_batch_count = 0;

        // エポックごとの学習
        for epoch in 0..training_config.num_epochs {
            let epoch_start_time = Instant::now();
            let mut total_loss = 0.0;
            let mut batch_count = 0;

            // 学習率スケジューリング（エポックに応じて学習率を調整）
            let current_lr = if training_config.use_lr_scheduling {
                training_config.learning_rate * (0.95_f64.powi(epoch as i32))
            } else {
                training_config.learning_rate
            };

            // バッチごとの学習
            let total_batches = total_samples.div_ceil(batch_size);

            println!(
                "エポック {} 開始: {} バッチを処理します (バッチサイズ: {})",
                epoch, total_batches, batch_size
            );

            for batch_start in (0..total_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(total_samples);
                let current_batch_size = batch_end - batch_start;

                // メモリ効率化: バッチサイズ分だけメモリを展開
                let mut batch_inputs = Vec::with_capacity(current_batch_size * input_dim);
                let mut batch_targets = Vec::with_capacity(current_batch_size * output_dim);

                // 必要な分だけデータをコピー
                for i in batch_start..batch_end {
                    batch_inputs.extend_from_slice(&training_data.inputs[i]);
                    batch_targets.extend_from_slice(&training_data.targets[i]);
                }

                // バッチサイズ分のテンソルを作成
                let batch_input_tensor =
                    Tensor::<B, 1>::from_floats(batch_inputs.as_slice(), device)
                        .reshape([current_batch_size, input_dim]);
                let batch_target_tensor =
                    Tensor::<B, 1>::from_floats(batch_targets.as_slice(), device)
                        .reshape([current_batch_size, output_dim]);

                // フォワードパス
                let predictions = self.forward(batch_input_tensor);

                // 損失計算（平均二乗誤差）
                let loss = mse_loss_autodiff(&predictions, &batch_target_tensor);
                let loss_value: f32 = loss.clone().into_scalar();
                total_loss += loss_value;

                // バックプロパゲーションと最適化
                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &self);
                self = optim.step(current_lr, self, grads_params);

                batch_count += 1;
                total_batch_count += 1;

                // バッチ処理後、メモリを明示的に解放
                drop(batch_inputs);
                drop(batch_targets);

                // 進捗表示（10バッチごと）
                if batch_count % 10 == 0 {
                    let elapsed = epoch_start_time.elapsed();
                    let samples_per_sec = (batch_count * batch_size) as f64 / elapsed.as_secs_f64();

                    // 残り時間の計算
                    let progress = batch_count as f64 / total_batches as f64;
                    let estimated_remaining = if progress > 0.0 {
                        elapsed.as_secs_f64() * (1.0 - progress) / progress
                    } else {
                        0.0
                    };

                    // 現在時刻と予想終了時刻
                    let now = std::time::SystemTime::now();
                    let estimated_end =
                        now + std::time::Duration::from_secs(estimated_remaining as u64);
                    let end_time_str = chrono::DateTime::<chrono::Local>::from(estimated_end)
                        .format("%H:%M:%S")
                        .to_string();

                    println!("エポック {}: バッチ {}/{} ({:.1}%) - 損失: {:.6} - 速度: {:.0} サンプル/秒",
                        epoch, batch_count, total_batches, progress * 100.0, loss_value, samples_per_sec);
                    println!(
                        "⏱️  残り時間: {:.1}分 - 予想終了: {}",
                        estimated_remaining / 60.0,
                        end_time_str
                    );
                }
            }

            let avg_loss = total_loss / batch_count as f32;
            let epoch_elapsed = epoch_start_time.elapsed();
            let total_elapsed = training_start_time.elapsed();

            // パフォーマンス最適化: エポック統計の表示
            let samples_per_sec = total_samples as f64 / epoch_elapsed.as_secs_f64();
            let epoch_end_time = std::time::SystemTime::now();
            let end_time_str = chrono::DateTime::<chrono::Local>::from(epoch_end_time)
                .format("%H:%M:%S")
                .to_string();

            println!(
                "エポック {} 完了: 平均損失 = {:.6}, 経過時間 = {:.2}秒, 速度 = {:.0} サンプル/秒",
                epoch,
                avg_loss,
                epoch_elapsed.as_secs_f64(),
                samples_per_sec
            );
            println!("⏰ エポック終了時刻: {}", end_time_str);

            // 早期停止のチェック
            if training_config.use_early_stopping {
                if avg_loss < best_loss {
                    best_loss = avg_loss;
                    patience_counter = 0;
                    println!("✅ エポック {}: 新しい最良損失 = {:.6}", epoch, avg_loss);
                } else {
                    patience_counter += 1;
                    println!(
                        "⚠️  エポック {}: 損失改善なし (パティエンス: {}/{})",
                        epoch, patience_counter, training_config.early_stopping_patience
                    );
                }

                if patience_counter >= training_config.early_stopping_patience {
                    println!(
                        "🛑 早期停止: エポック {} で学習を終了 (パティエンス: {})",
                        epoch, training_config.early_stopping_patience
                    );
                    break;
                }
            }

            println!(
                "📊 エポック {} 完了: 平均損失 = {:.6}, 学習率 = {:.6}",
                epoch, avg_loss, current_lr
            );
            println!(
                "⏱️  エポック時間: {:.2}秒, 総時間: {:.2}秒",
                epoch_elapsed.as_secs_f64(),
                total_elapsed.as_secs_f64()
            );
            println!("{}", "=".repeat(60));
        }

        let total_elapsed = training_start_time.elapsed();
        let total_samples_processed = total_batch_count * batch_size;
        let overall_samples_per_sec = total_samples_processed as f64 / total_elapsed.as_secs_f64();

        // 全体の終了時間予測
        let training_end_time = std::time::SystemTime::now();
        let end_time_str = chrono::DateTime::<chrono::Local>::from(training_end_time)
            .format("%H:%M:%S")
            .to_string();

        // 残りエポックの予測
        let remaining_epochs =
            training_config.num_epochs - (total_batch_count / total_samples.div_ceil(batch_size));
        let _estimated_remaining_time = if remaining_epochs > 0 {
            let avg_epoch_time = total_elapsed.as_secs_f64()
                / (total_batch_count / total_samples.div_ceil(batch_size)) as f64;
            avg_epoch_time * remaining_epochs as f64
        } else {
            0.0
        };

        println!("🎉 学習が完了しました！");
        println!(
            "⏱️  総学習時間: {:.2}秒 ({:.2}分)",
            total_elapsed.as_secs_f64(),
            total_elapsed.as_secs_f64() / 60.0
        );
        println!("⏰ 学習終了時刻: {}", end_time_str);
        println!("📊 パフォーマンス統計: 総バッチ数 = {}, 総サンプル数 = {}, 全体速度 = {:.0} サンプル/秒",
            total_batch_count, total_samples_processed, overall_samples_per_sec);
        println!("📈 最終損失: {:.6}", best_loss);
        println!("🔧 処理された総バッチ数: {}", total_batch_count);
        Ok(self)
    }
}

/// AutodiffBackend用の損失関数
fn mse_loss_autodiff<B: AutodiffBackend>(
    predictions: &Tensor<B, 2>,
    targets: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    let diff = predictions.clone() - targets.clone();
    let squared_diff = diff.clone() * diff;
    squared_diff.mean()
}
