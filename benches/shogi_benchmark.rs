use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustshogi::bitboard::BitBoard;

fn benchmark_bitboard_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("BitBoard Operations");

    let bb1 = black_box(BitBoard::from_u128(1124249833570304));
    let bb2 = black_box(BitBoard::from_u128(548949983232));

    group.bench_function("bitand", |b| {
        b.iter(|| {
            let _ = bb1 & bb2;
        });
    });

    group.bench_function("bitor", |b| {
        b.iter(|| {
            let _ = bb1 | bb2;
        });
    });

    group.bench_function("get_trues", |b| {
        b.iter(|| {
            let _ = bb1.get_trues();
        });
    });

    // Batch operations benchmarks
    let boards: Vec<BitBoard> = (0..10).map(|i| BitBoard::from_u128(1u128 << i)).collect();
    let boards_slice = black_box(&boards);

    group.bench_function("bitand_batch", |b| {
        b.iter(|| {
            let _ = BitBoard::bitand_batch(boards_slice);
        });
    });

    group.bench_function("bitor_batch", |b| {
        b.iter(|| {
            let _ = BitBoard::bitor_batch(boards_slice);
        });
    });

    group.bench_function("bitxor_batch", |b| {
        b.iter(|| {
            let _ = BitBoard::bitxor_batch(boards_slice);
        });
    });

    // count_ones benchmark
    let bb_max = black_box(BitBoard::from_u128(u128::MAX));
    group.bench_function("count_ones", |b| {
        b.iter(|| {
            let _ = bb_max.count_ones();
        });
    });

    // Batch shift benchmarks
    let batch_boards: Vec<BitBoard> = (0..100)
        .map(|i| BitBoard::from_u128(u128::wrapping_add(u128::MAX / (i + 1), i)))
        .collect();
    let batch_boards_slice = black_box(&batch_boards);

    for &shift_amount in &[1, 33, 65] {
        // Scalar shift benchmarks
        group.bench_function(format!("shift_right_batch_scalar_{}", shift_amount), |b| {
            b.iter(|| {
                let _ = batch_boards_slice
                    .iter()
                    .map(|board| *board >> shift_amount)
                    .collect::<Vec<_>>();
            });
        });
        group.bench_function(format!("shift_left_batch_scalar_{}", shift_amount), |b| {
            b.iter(|| {
                let _ = batch_boards_slice
                    .iter()
                    .map(|board| *board << shift_amount)
                    .collect::<Vec<_>>();
            });
        });

        // AVX2 shift benchmarks
        if cfg!(target_feature = "avx2") {
            group.bench_function(format!("shift_right_batch_avx2_{}", shift_amount), |b| {
                b.iter(|| {
                    let _ = BitBoard::shift_right_batch(batch_boards_slice, shift_amount);
                });
            });
            group.bench_function(format!("shift_left_batch_avx2_{}", shift_amount), |b| {
                b.iter(|| {
                    let _ = BitBoard::shift_left_batch(batch_boards_slice, shift_amount);
                });
            });
        }
    }

    group.finish();
}

fn benchmark_game_logic(c: &mut Criterion) {
    use rustshogi::color::ColorType;
    use rustshogi::game::Game;

    let mut group = c.benchmark_group("Game Logic");

    let mut game = Game::new();
    game.input_board("startpos".to_string());
    let board = black_box(game.board);

    group.bench_function("search_moves", |b| {
        b.iter(|| {
            let _ = board.search_moves(ColorType::Black);
        });
    });

    group.bench_function("random_game", |b| {
        b.iter(|| {
            let mut game = Game::new();
            game.input_board("startpos".to_string());
            let _result_game = game.one_play();
        });
    });

    group.finish();
}

fn benchmark_direction(c: &mut Criterion) {
    use rustshogi::direction::Direction;
    let mut group = c.benchmark_group("direction");
    group.bench_function("get_all_direction_vectors", |b| {
        b.iter(|| {
            let _ = Direction::get_all_direction_vectors();
        })
    });
    if cfg!(all(target_arch = "x86_64", target_feature = "sse2")) {
        group.bench_function("get_all_direction_vectors_simd", |b| {
            b.iter(|| unsafe {
                let _ = Direction::get_all_direction_vectors_simd();
            })
        });
    }
    group.finish();
}

fn benchmark_piece(c: &mut Criterion) {
    use rustshogi::piece::{Piece, PieceType};
    let mut group = c.benchmark_group("piece");
    let piece_types = [
        PieceType::Rook,
        PieceType::Bichop,
        PieceType::Silver,
        PieceType::Knight,
        PieceType::Lance,
        PieceType::Pawn,
        PieceType::King,
        PieceType::Gold,
        PieceType::Rook,
        PieceType::Bichop,
        PieceType::Silver,
        PieceType::Knight,
        PieceType::Lance,
        PieceType::Pawn,
        PieceType::King,
        PieceType::Gold,
    ];

    group.bench_function("able_pro_batch", |b| {
        b.iter(|| {
            let _ = Piece::able_pro_batch(&piece_types);
        })
    });

    if cfg!(all(target_arch = "x86_64", target_feature = "sse2")) {
        group.bench_function("able_pro_batch_simd", |b| {
            b.iter(|| unsafe {
                let _ = Piece::able_pro_batch_simd(&piece_types);
            })
        });
    }
    group.finish();
}

fn benchmark_random(c: &mut Criterion) {
    use rustshogi::random::Random;
    let mut group = c.benchmark_group("random");
    let mut rng = Random::new(1, 100);
    let len: u16 = 1024;

    group.bench_function("generate_multi", |b| {
        b.iter(|| {
            let _ = rng.generate_multi(len);
        })
    });

    group.bench_function("generate_multi_fast", |b| {
        b.iter(|| {
            let _ = rng.generate_multi_fast(len);
        })
    });

    if cfg!(all(target_arch = "x86_64", target_feature = "sse2")) {
        group.bench_function("generate_multi_sse2", |b| {
            b.iter(|| unsafe {
                let _ = rng.generate_multi_sse2(len);
            })
        });
    }
    group.finish();
}

fn benchmark_evaluator(c: &mut Criterion) {
    use rustshogi::evaluator::{DatabaseType, Evaluator};
    use std::fs;

    let db_path = "test_benchmark.db";
    let db_type = DatabaseType::Sqlite(db_path.to_string());
    let evaluator = Evaluator::new(db_type);

    // Setup: データベースを初期化し、ランダムな盤面をいくつか生成
    evaluator.init_database().unwrap();
    evaluator.generate_and_save_random_boards(10).unwrap();

    let mut group = c.benchmark_group("Evaluator");

    // ベンチマーク：順次実行
    group.bench_function("update_records_sequential", |b| {
        b.iter(|| {
            let _ = evaluator.update_records_with_random_games(
                black_box(10), // trials_per_record
                black_box(Some(10)), // max_records
                black_box(1), // num_threads (sequential)
            );
        })
    });

    // ベンチマーク：並列実行
    group.bench_function("update_records_parallel", |b| {
        b.iter(|| {
            let _ = evaluator.update_records_with_random_games_parallel(
                black_box(10), // trials_per_record
                black_box(Some(10)), // max_records
                black_box(num_cpus::get()), // num_threads
            );
        })
    });

    group.finish();

    // Teardown: テストデータベースファイルを削除
    fs::remove_file(db_path).unwrap();
}

criterion_group!(
    benches,
    benchmark_bitboard_operations,
    benchmark_game_logic,
    benchmark_direction,
    benchmark_piece,
    benchmark_random,
    benchmark_evaluator
);
criterion_main!(benches);
