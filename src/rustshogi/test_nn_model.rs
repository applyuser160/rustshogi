use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};

use crate::nn_model::NnModel;

use super::board::Board;
use super::nn_model::{NnModelConfig, TrainingConfig, TrainingData};

#[test]
fn test_nn_model_config() {
    let config = NnModelConfig::default();

    // 設定が正しいことを確認
    assert_eq!(config.input_dim, 2320);
    assert_eq!(config.output_dim, 3);
    assert_eq!(config.hidden_dims, vec![512, 256, 128]);
    assert_eq!(config.dropout_rate, 0.1);
}

#[test]
fn test_training_config() {
    let config = TrainingConfig::default();

    // 学習設定が正しいことを確認
    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.batch_size, 32);
    assert_eq!(config.num_epochs, 100);
    assert_eq!(config.model_save_path, "model.bin");
}

#[test]
fn test_training_data() {
    let mut training_data = TrainingData::new();

    // 空のデータを確認
    assert!(training_data.is_empty());
    assert_eq!(training_data.len(), 0);

    // サンプルデータを追加
    let input = vec![0.0; 2320];
    let target = vec![1.0, 2.0, 3.0];
    training_data.add_sample(input, target);

    // データが追加されたことを確認
    assert!(!training_data.is_empty());
    assert_eq!(training_data.len(), 1);
    assert_eq!(training_data.inputs[0].len(), 2320);
    assert_eq!(training_data.targets[0].len(), 3);
}

#[test]
fn test_board_to_vector_output_size() {
    let board = Board::new();
    let vector = board.to_vector(None);

    // board.to_vectorの出力が2320次元であることを確認
    assert_eq!(vector.len(), 2320);
}

#[test]
fn test_board_startpos_to_vector() {
    let mut board = Board::new();
    board.startpos();
    let vector = board.to_vector(None);

    // 開始局面のベクターが正しいサイズであることを確認
    assert_eq!(vector.len(), 2320);

    // ベクターに非ゼロ値が含まれることを確認（開始局面なので駒が配置されている）
    let has_non_zero = vector.iter().any(|&x| x != 0.0);
    assert!(has_non_zero);
}

#[test]
fn test_model_save_load() {
    use std::fs;
    use std::path::Path;

    // デバイスを明示的に指定しないテスト
    let test_path = "test_model.json";

    // ダミーのモデルデータを作成して保存
    let save_data = super::nn_model::ModelSaveData {
        config: NnModelConfig::default(),
        hidden_layers_weights: vec![
            vec![vec![1.0; 2320]; 512], // 入力層 -> 隠れ層1
            vec![vec![1.0; 512]; 256],  // 隠れ層1 -> 隠れ層2
            vec![vec![1.0; 256]; 128],  // 隠れ層2 -> 隠れ層3
        ],
        hidden_layers_bias: vec![
            vec![0.0; 512], // 隠れ層1のバイアス
            vec![0.0; 256], // 隠れ層2のバイアス
            vec![0.0; 128], // 隠れ層3のバイアス
        ],
        output_layer_weights: vec![vec![1.0; 128]; 3],
        output_layer_bias: vec![0.0; 3],
    };

    // JSON形式で保存
    let json_data = serde_json::to_string_pretty(&save_data).unwrap();
    fs::write(test_path, json_data).unwrap();

    // ファイルが作成されたことを確認
    assert!(Path::new(test_path).exists());

    // ファイルの内容を確認
    let contents = fs::read_to_string(test_path).unwrap();
    assert!(contents.contains("hidden_layers_weights"));
    assert!(contents.contains("output_layer_weights"));

    // ファイルを読み込み
    let loaded_json = fs::read_to_string(test_path).unwrap();
    let loaded_data: super::nn_model::ModelSaveData = serde_json::from_str(&loaded_json).unwrap();
    assert_eq!(loaded_data.hidden_layers_weights.len(), 3);

    // テストファイルを削除
    let _ = fs::remove_file(test_path);
}

#[test]
fn test_model_weights_access() {
    // 重みデータの構造をテスト
    let weights = super::nn_model::ModelSaveData {
        config: NnModelConfig::default(),
        hidden_layers_weights: vec![
            vec![vec![0.0; 2320]; 512], // 入力層 -> 隠れ層1
            vec![vec![0.0; 512]; 256],  // 隠れ層1 -> 隠れ層2
            vec![vec![0.0; 256]; 128],  // 隠れ層2 -> 隠れ層3
        ],
        hidden_layers_bias: vec![
            vec![0.0; 512], // 隠れ層1のバイアス
            vec![0.0; 256], // 隠れ層2のバイアス
            vec![0.0; 128], // 隠れ層3のバイアス
        ],
        output_layer_weights: vec![vec![0.0; 128]; 3],
        output_layer_bias: vec![0.0; 3],
    };

    // 重みの構造を確認
    assert_eq!(weights.hidden_layers_weights.len(), 3);
    assert_eq!(weights.hidden_layers_weights[0].len(), 512);
    assert_eq!(weights.hidden_layers_weights[0][0].len(), 2320);
    assert_eq!(weights.hidden_layers_bias.len(), 3);
    assert_eq!(weights.hidden_layers_bias[0].len(), 512);
    assert_eq!(weights.output_layer_weights.len(), 3);
    assert_eq!(weights.output_layer_weights[0].len(), 128);
    assert_eq!(weights.output_layer_bias.len(), 3);
}

#[test]
fn test_training_with_optimization() {
    use super::board::Board;
    use super::nn_model::{NnModelConfig, TrainingConfig, TrainingData};

    // テスト用の学習データを作成
    let mut training_data = TrainingData::new();

    // 開始局面のデータを追加
    let mut board = Board::new();
    board.startpos();
    let vector = board.to_vector(None);

    // ダミーのMCTS結果を作成
    let white_wins = 100.0;
    let black_wins = 80.0;
    let total_games = 200.0;

    training_data.add_sample(vector, vec![white_wins, black_wins, total_games]);

    // 学習設定
    let training_config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 1,
        num_epochs: 5, // 短いエポック数でテスト
        model_save_path: "test_model.bin".to_string(),
    };

    // モデルを作成（デバイスを明示的に指定しない）
    let config = NnModelConfig::default();
    let device = NdArrayDevice::Cpu;
    let model = NnModel::<Autodiff<NdArray>>::new(&config, &device);

    // 学習を実行
    let trained_model = model.train(&training_data, &training_config, &device);
    assert!(trained_model.is_ok());

    // 学習データの構造をテスト
    assert_eq!(training_data.len(), 1);
    assert_eq!(training_data.inputs[0].len(), 2320);
    assert_eq!(training_data.targets[0].len(), 3);

    println!("学習テストが完了しました");
}

#[test]
fn test_training_full_with_autodiff() {
    use super::board::Board;
    use super::nn_model::{NnModel, NnModelConfig, TrainingConfig, TrainingData};

    // テスト用の学習データを作成
    let mut training_data = TrainingData::new();

    // 複数の局面のデータを追加
    for i in 0..3 {
        let mut board = Board::new();
        board.startpos();
        let vector = board.to_vector(None);

        let white_wins = 100.0 + i as f32 * 10.0;
        let black_wins = 80.0 + i as f32 * 5.0;
        let total_games = 200.0 + i as f32 * 15.0;

        training_data.add_sample(vector, vec![white_wins, black_wins, total_games]);
    }

    // 学習設定
    let training_config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 2,
        num_epochs: 3, // 短いエポック数でテスト
        model_save_path: "test_model_full.bin".to_string(),
    };

    // モデルを作成（デバイスを明示的に指定しない）
    let config = NnModelConfig::default();
    let device = NdArrayDevice::Cpu;
    let model = NnModel::<Autodiff<NdArray>>::new(&config, &device);

    // AutodiffBackend用の完全な学習を実行
    let trained_model = model.train(&training_data, &training_config, &device);
    assert!(trained_model.is_ok());

    // 学習データの構造をテスト
    assert_eq!(training_data.len(), 3);
    for i in 0..3 {
        assert_eq!(training_data.inputs[i].len(), 2320);
        assert_eq!(training_data.targets[i].len(), 3);
    }

    println!("AutodiffBackend用の完全な学習テストが完了しました");
}
