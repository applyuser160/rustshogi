use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use rustshogi::board::Board;
use rustshogi::nn_model::NnModel;
use rustshogi::nn_model::{ModelSaveData, NnModelConfig, TrainingConfig, TrainingData};

#[test]
fn test_nn_model_config() {
    let config = NnModelConfig::default();

    // Check that the settings are correct
    assert_eq!(config.input_dim, 2320);
    assert_eq!(config.output_dim, 3);
    assert_eq!(config.hidden_dims, vec![1024, 512, 256]); // 3 hidden layers
    assert_eq!(config.dropout_rate, 0.3);
}

#[test]
fn test_training_config() {
    let config = TrainingConfig::default();

    // Check that the training settings are correct
    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.batch_size, 64); // Default value changed to 64
    assert_eq!(config.num_epochs, 100);
    assert_eq!(config.model_save_path, "model.bin");
    assert_eq!(config.use_lr_scheduling, true);
    assert_eq!(config.use_early_stopping, true);
    assert_eq!(config.early_stopping_patience, 10);
}

#[test]
fn test_training_data() {
    let mut training_data = TrainingData::new();

    // Check empty data
    assert!(training_data.is_empty());
    assert_eq!(training_data.len(), 0);

    // Add sample data
    let input = vec![0.0; 2320];
    let target = vec![1.0, 2.0, 3.0];
    training_data.add_sample(input, target);

    // Check that data has been added
    assert!(!training_data.is_empty());
    assert_eq!(training_data.len(), 1);
    assert_eq!(training_data.inputs[0].len(), 2320);
    assert_eq!(training_data.targets[0].len(), 3);
}

#[test]
fn test_board_to_vector_output_size() {
    let board = Board::new();
    let vector = board.to_vector(None);

    // Check that the output of board.to_vector is 2320 dimensions
    assert_eq!(vector.len(), 2320);
}

#[test]
fn test_board_startpos_to_vector() {
    let mut board = Board::new();
    board.startpos();
    let vector = board.to_vector(None);

    // Check that the vector for the starting position is the correct size
    assert_eq!(vector.len(), 2320);

    // Check that the vector contains non-zero values (since pieces are placed in the starting position)
    let has_non_zero = vector.iter().any(|&x| x != 0.0);
    assert!(has_non_zero);
}

#[test]
fn test_model_save_load() {
    use std::fs;
    use std::path::Path;

    // Test without explicitly specifying the device
    let test_path = "test_model.json";

    // Create and save dummy model data
    let save_data = ModelSaveData {
        config: NnModelConfig::default(),
        hidden_layers_weights: vec![
            vec![vec![1.0; 2320]; 1024], // Input layer -> Hidden layer 1
            vec![vec![1.0; 1024]; 512],  // Hidden layer 1 -> Hidden layer 2
            vec![vec![1.0; 512]; 256],   // Hidden layer 2 -> Hidden layer 3
        ],
        hidden_layers_bias: vec![
            vec![0.0; 1024], // Bias for hidden layer 1
            vec![0.0; 512],  // Bias for hidden layer 2
            vec![0.0; 256],  // Bias for hidden layer 3
        ],
        output_layer_weights: vec![vec![1.0; 256]; 3],
        output_layer_bias: vec![0.0; 3],
    };

    // Save in JSON format
    let json_data = serde_json::to_string_pretty(&save_data).unwrap();
    fs::write(test_path, json_data).unwrap();

    // Check that the file was created
    assert!(Path::new(test_path).exists());

    // Check the file content
    let contents = fs::read_to_string(test_path).unwrap();
    assert!(contents.contains("hidden_layers_weights"));
    assert!(contents.contains("output_layer_weights"));

    // Read the file
    let loaded_json = fs::read_to_string(test_path).unwrap();
    let loaded_data: ModelSaveData = serde_json::from_str(&loaded_json).unwrap();
    assert_eq!(loaded_data.hidden_layers_weights.len(), 3);

    // Delete the test file
    let _ = fs::remove_file(test_path);
}

#[test]
fn test_model_weights_access() {
    // Test the structure of the weight data
    let weights = ModelSaveData {
        config: NnModelConfig::default(),
        hidden_layers_weights: vec![
            vec![vec![0.0; 2320]; 1024], // Input layer -> Hidden layer 1
            vec![vec![0.0; 1024]; 512],  // Hidden layer 1 -> Hidden layer 2
            vec![vec![0.0; 512]; 256],   // Hidden layer 2 -> Hidden layer 3
        ],
        hidden_layers_bias: vec![
            vec![0.0; 1024], // Bias for hidden layer 1
            vec![0.0; 512],  // Bias for hidden layer 2
            vec![0.0; 256],  // Bias for hidden layer 3
        ],
        output_layer_weights: vec![vec![0.0; 256]; 3],
        output_layer_bias: vec![0.0; 3],
    };

    // Check the structure of the weights
    assert_eq!(weights.hidden_layers_weights.len(), 3);
    assert_eq!(weights.hidden_layers_weights[0].len(), 1024);
    assert_eq!(weights.hidden_layers_weights[0][0].len(), 2320);
    assert_eq!(weights.hidden_layers_bias.len(), 3);
    assert_eq!(weights.hidden_layers_bias[0].len(), 1024);
    assert_eq!(weights.output_layer_weights.len(), 3);
    assert_eq!(weights.output_layer_weights[0].len(), 256);
    assert_eq!(weights.output_layer_bias.len(), 3);
}

#[test]
fn test_training_with_optimization() {
    use rustshogi::board::Board;
    use rustshogi::nn_model::{NnModelConfig, TrainingConfig, TrainingData};

    // Create test training data
    let mut training_data = TrainingData::new();

    // Add data for the starting position
    let mut board = Board::new();
    board.startpos();
    let vector = board.to_vector(None);

    // Create dummy MCTS results
    let white_wins = 100.0;
    let black_wins = 80.0;
    let total_games = 200.0;

    training_data.add_sample(vector, vec![white_wins, black_wins, total_games]);

    // Training settings
    let training_config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 1,
        num_epochs: 5, // Test with a small number of epochs
        model_save_path: "test_model.bin".to_string(),
        use_lr_scheduling: true,
        use_early_stopping: true,
        early_stopping_patience: 10,
    };

    // Create the model (without explicitly specifying the device)
    let config = NnModelConfig::default();
    let device = NdArrayDevice::Cpu;
    let model = NnModel::<Autodiff<NdArray>>::new(&config, &device);

    // Run training
    let trained_model = model.train(&training_data, &training_config, &device);
    assert!(trained_model.is_ok());

    // Test the structure of the training data
    assert_eq!(training_data.len(), 1);
    assert_eq!(training_data.inputs[0].len(), 2320);
    assert_eq!(training_data.targets[0].len(), 3);

    println!("Training test completed");
}

#[test]
fn test_training_full_with_autodiff() {
    use rustshogi::board::Board;
    use rustshogi::nn_model::{NnModel, NnModelConfig, TrainingConfig, TrainingData};

    // Create test training data
    let mut training_data = TrainingData::new();

    // Add data for multiple positions
    for i in 0..3 {
        let mut board = Board::new();
        board.startpos();
        let vector = board.to_vector(None);

        let white_wins = 100.0 + i as f32 * 10.0;
        let black_wins = 80.0 + i as f32 * 5.0;
        let total_games = 200.0 + i as f32 * 15.0;

        training_data.add_sample(vector, vec![white_wins, black_wins, total_games]);
    }

    // Training settings
    let training_config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 2,
        num_epochs: 3, // Test with a small number of epochs
        model_save_path: "test_model_full.bin".to_string(),
        use_lr_scheduling: true,
        use_early_stopping: true,
        early_stopping_patience: 10,
    };

    // Create the model (without explicitly specifying the device)
    let config = NnModelConfig::default();
    let device = NdArrayDevice::Cpu;
    let model = NnModel::<Autodiff<NdArray>>::new(&config, &device);

    // Run full training for AutodiffBackend
    let trained_model = model.train(&training_data, &training_config, &device);
    assert!(trained_model.is_ok());

    // Test the structure of the training data
    assert_eq!(training_data.len(), 3);
    for i in 0..3 {
        assert_eq!(training_data.inputs[i].len(), 2320);
        assert_eq!(training_data.targets[i].len(), 3);
    }

    println!("Full training test for AutodiffBackend completed");
}
