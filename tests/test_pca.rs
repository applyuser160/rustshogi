use rustshogi::pca::*;
use ndarray::{Array1, Array2};

#[test]
fn test_pca_transform_new() {
    let components = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let mean = Array1::from_vec(vec![0.5, 1.0, 1.5]);
    let pca_transform = PCATransform::new(components, mean, 2);

    assert_eq!(pca_transform.n_components, 2);
    assert_eq!(pca_transform.mean.len(), 3);
    assert_eq!(pca_transform.components.shape(), [2, 3]);
}

#[test]
fn test_pca_transform_transform() {
    let components = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let mean = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let pca_transform = PCATransform::new(components, mean, 2);

    let features = vec![1.0, 2.0, 3.0];
    let transformed = pca_transform.transform(&features);

    assert_eq!(transformed.len(), 2);
    assert_eq!(transformed[0], 1.0);
    assert_eq!(transformed[1], 2.0);
}

#[test]
fn test_learn_simple_pca() {
    let samples = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 3.0, 4.0],
        vec![3.0, 4.0, 5.0],
    ];

    let pca_transform = learn_simple_pca(&samples, 2);

    assert_eq!(pca_transform.n_components, 2);
    assert_eq!(pca_transform.mean.len(), 3);
    assert_eq!(pca_transform.components.shape(), [2, 3]);
}

#[test]
fn test_learn_pca() {
    let samples = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
        vec![4.0, 5.0],
    ];

    let result = learn_pca(&samples, 2);
    assert!(result.is_ok());

    let pca_transform = result.unwrap();
    assert_eq!(pca_transform.n_components, 2);
    assert_eq!(pca_transform.mean.len(), 2);
    assert_eq!(pca_transform.components.shape(), [2, 2]);
}

#[test]
fn test_learn_pca_empty_samples() {
    let samples: Vec<Vec<f32>> = vec![];
    let result = learn_pca(&samples, 2);
    assert!(result.is_err());
}

#[test]
fn test_learn_pca_too_many_components() {
    let samples = vec![vec![1.0, 2.0]];
    let result = learn_pca(&samples, 3);
    assert!(result.is_err());
}

#[test]
fn test_apply_pca_compression_no_transform() {
    let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let compressed = apply_pca_compression(&features, 3);

    assert_eq!(compressed.len(), 3);
}

#[test]
fn test_apply_pca_compression_target_dims_larger() {
    let features = vec![1.0, 2.0, 3.0];
    let compressed = apply_pca_compression(&features, 5);

    assert_eq!(compressed.len(), 3);
    assert_eq!(compressed, features);
}

#[test]
fn test_set_and_get_global_pca_transform() {
    let components = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let mean = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let pca_transform = PCATransform::new(components, mean, 2);

    set_global_pca_transform(pca_transform.clone());
    let retrieved = get_global_pca_transform();

    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), pca_transform);
}
