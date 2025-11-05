use nalgebra::DMatrix;
use ndarray::{Array1, Array2, Axis};
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Structure to store the PCA transformation matrix (using ndarray)
#[derive(Clone, Debug, PartialEq)]
pub struct PCATransform {
    pub components: Array2<f32>, // Principal component matrix
    pub mean: Array1<f32>,       // Mean vector
    pub n_components: usize,     // Number of principal components
}

impl PCATransform {
    /// Create a new PCA transformation
    pub fn new(components: Array2<f32>, mean: Array1<f32>, n_components: usize) -> Self {
        Self {
            components,
            mean,
            n_components,
        }
    }

    /// Transform features
    pub fn transform(&self, features: &[f32]) -> Vec<f32> {
        if features.len() != self.mean.len() {
            panic!(
                "Feature dimension mismatch: expected {}, got {}",
                self.mean.len(),
                features.len()
            );
        }

        // Convert features to ndarray
        let data = Array1::from_vec(features.to_vec());

        // Subtract the mean
        let centered = &data - &self.mean;

        // Apply principal components
        let transformed = self.components.dot(&centered);

        transformed.to_vec()
    }
}

// Store the global PCA transformation
static PCA_TRANSFORM: Lazy<Mutex<Option<PCATransform>>> = Lazy::new(|| Mutex::new(None));

/// Set the global PCA transformation
pub fn set_global_pca_transform(pca_transform: PCATransform) {
    if let Ok(mut transform) = PCA_TRANSFORM.lock() {
        *transform = Some(pca_transform);
    }
}

/// Get the global PCA transformation
pub fn get_global_pca_transform() -> Option<PCATransform> {
    if let Ok(transform) = PCA_TRANSFORM.lock() {
        transform.clone()
    } else {
        None
    }
}

/// Full PCA learning (using nalgebra)
pub fn learn_pca(samples: &[Vec<f32>], n_components: usize) -> Result<PCATransform, String> {
    if samples.is_empty() {
        return Err("Cannot learn PCA from empty samples".to_string());
    }

    let n_samples = samples.len();
    let n_features = samples[0].len();

    if n_components > n_features {
        return Err(format!(
            "Number of components ({}) cannot be greater than number of features ({})",
            n_components, n_features
        ));
    }

    // Convert data to ndarray
    let mut data = Array2::zeros((n_samples, n_features));
    for (i, sample) in samples.iter().enumerate() {
        for (j, &value) in sample.iter().enumerate() {
            data[[i, j]] = value;
        }
    }

    // Calculate the mean
    let mean = data.mean_axis(Axis(0)).unwrap();

    // Center the data (subtract the mean)
    let mut centered = data.clone();
    for mut row in centered.rows_mut() {
        row -= &mean;
    }

    // Convert to nalgebra's DMatrix
    let mut nalgebra_data = DMatrix::zeros(n_samples, n_features);
    for i in 0..n_samples {
        for j in 0..n_features {
            nalgebra_data[(i, j)] = centered[[i, j]];
        }
    }

    // Calculate the covariance matrix: C = (X^T * X) / (n-1)
    let covariance = nalgebra_data.transpose() * &nalgebra_data / (n_samples - 1) as f32;

    // Perform eigenvalue decomposition (using nalgebra's symmetric eigenvalue decomposition)
    let eigen = match covariance.symmetric_eigen() {
        eigen => eigen,
    };

    // Get eigenvalues and eigenvectors
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Sort by eigenvalue in descending order
    let mut eigenval_vec: Vec<(usize, f32)> = Vec::new();
    for i in 0..eigenvalues.len() {
        eigenval_vec.push((i, eigenvalues[i]));
    }

    eigenval_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Select the top n_components principal components
    let mut components = Array2::zeros((n_components, n_features));
    for i in 0..n_components {
        if i < eigenval_vec.len() {
            let idx = eigenval_vec[i].0;
            for j in 0..n_features {
                components[[i, j]] = eigenvectors[(j, idx)];
            }
        }
    }

    Ok(PCATransform::new(components, mean, n_components))
}

/// Simple PCA learning (variance-based selection)
pub fn learn_simple_pca(samples: &[Vec<f32>], n_components: usize) -> PCATransform {
    if samples.is_empty() {
        panic!("Cannot learn PCA from empty samples");
    }

    let n_features = samples[0].len();

    // Convert data to ndarray
    let mut data = Array2::zeros((samples.len(), n_features));
    for (i, sample) in samples.iter().enumerate() {
        for (j, &value) in sample.iter().enumerate() {
            data[[i, j]] = value;
        }
    }

    // Calculate the mean
    let mean = data.mean_axis(Axis(0)).unwrap();

    // Calculate the variance
    let mut variances = Array1::zeros(n_features);
    for row in data.rows() {
        let centered_row = &row - &mean;
        for (i, &value) in centered_row.iter().enumerate() {
            variances[i] += value * value;
        }
    }
    variances /= samples.len() as f32;

    // Sort indices by variance in descending order
    let mut indices: Vec<usize> = (0..n_features).collect();
    indices.sort_by(|&a, &b| variances[b].partial_cmp(&variances[a]).unwrap());

    // Create principal components (simple implementation: unit vectors)
    let mut components = Array2::zeros((n_components, n_features));
    for i in 0..n_components {
        if i < indices.len() {
            components[[i, indices[i]]] = 1.0;
        }
    }

    PCATransform::new(components, mean, n_components)
}

/// Apply dimensionality reduction by PCA
pub fn apply_pca_compression(features: &[f32], target_dims: usize) -> Vec<f32> {
    if target_dims >= features.len() {
        return features.to_vec();
    }

    // If a global PCA transformation is available, use it
    if let Some(ref transform) = get_global_pca_transform() {
        if transform.n_components == target_dims {
            return transform.transform(features);
        }
    }

    // If no PCA transformation is available, use simple sampling
    let mut compressed = Vec::with_capacity(target_dims);
    let step = features.len() as f32 / target_dims as f32;
    for i in 0..target_dims {
        let index = (i as f32 * step) as usize;
        compressed.push(features[index]);
    }

    compressed
}
