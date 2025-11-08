#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign,
};

pub const LENGTH_OF_BOARD: u8 = 121;

pub const LENGTH_OF_FRAME: u8 = 11;

pub const LENGTH_OF_EDGE: u8 = 9;

pub const BIT_OF_FRAME: u128 = 340199411925109678410730024455729840000;

pub const STRING_OF_FRAME: &str = "\
11111111111\
10000000001\
10000000001\
10000000001\
10000000001\
10000000001\
10000000001\
10000000001\
10000000001\
10000000001\
11111111111";

pub const BIT_OF_PRO_ZONE_BLACK: u128 = 1124249833570304;

pub const STRING_OF_PRO_ZONE_BLACK: &str = "\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
01111111110\
01111111110\
01111111110\
00000000000";

pub const BIT_OF_PRO_ZONE_WHITE: u128 = 82954995819127820108022929011245056;

pub const STRING_OF_PRO_ZONE_WHITE: &str = "\
00000000000\
01111111110\
01111111110\
01111111110\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000";

pub const BIT_OF_LAST1_ZONE_BLACK: u128 = 267911168;

pub const STRING_OF_LAST1_ZONE_BLACK: &str = "\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
01111111110\
00000000000";

pub const BIT_OF_LAST1_ZONE_WHITE: u128 = 82914490459728028693096363257233408;

pub const STRING_OF_LAST1_ZONE_WHITE: &str = "\
00000000000\
01111111110\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000";

pub const BIT_OF_LAST2_ZONE_BLACK: u128 = 548949983232;

pub const STRING_OF_LAST2_ZONE_BLACK: &str = "\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
01111111110\
01111111110\
00000000000";

pub const BIT_OF_LAST2_ZONE_WHITE: u128 = 82954976050772817769606664215855104;

pub const STRING_OF_LAST2_ZONE_WHITE: &str = "\
00000000000\
01111111110\
01111111110\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000\
00000000000";

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BitBoard {
    // data[0] is MSB, data[1] is LSB
    data: [u64; 2],
}

impl Default for BitBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl BitBoard {
    pub fn new() -> Self {
        BitBoard { data: [0, 0] }
    }

    pub fn from_bitboard(bitboard: Self) -> Self {
        bitboard
    }

    pub fn from_u128(integer: u128) -> Self {
        BitBoard {
            data: [(integer >> 64) as u64, integer as u64],
        }
    }

    pub fn from_str(string: &str) -> Self {
        let mut res: u128 = 0u128;
        let mut bit_pos: i32 = 127;
        for c in string.chars() {
            if bit_pos < 128 - LENGTH_OF_BOARD as i32 {
                break;
            }
            if c == '1' {
                res |= 1u128 << bit_pos;
                bit_pos -= 1;
            } else if c == '0' {
                bit_pos -= 1;
            }
        }
        Self::from_u128(res)
    }

    pub fn to_u128(&self) -> u128 {
        ((self.data[0] as u128) << 64) | (self.data[1] as u128)
    }

    pub fn get_trues(&self) -> Vec<u8> {
        let mut result: Vec<u8> = Vec::new();
        let mut d0: u64 = self.data[0];
        while d0 != 0 {
            let index: u8 = d0.leading_zeros() as u8;
            result.push(index);
            d0 &= !(1u64 << (63 - index));
        }
        let mut d1: u64 = self.data[1];
        while d1 != 0 {
            let index: u8 = d1.leading_zeros() as u8;
            result.push(index + 64);
            d1 &= !(1u64 << (63 - index));
        }
        result
    }

    pub fn get_trues_iter(&self) -> impl Iterator<Item = u8> + '_ {
        let mut d0: u64 = self.data[0];
        let mut d1: u64 = self.data[1];
        let mut state: i32 = 0; // 0 for d0, 1 for d1

        std::iter::from_fn(move || {
            if state == 0 {
                if d0 == 0 {
                    state = 1;
                } else {
                    let index: u8 = d0.leading_zeros() as u8;
                    d0 &= !(1u64 << (63 - index));
                    return Some(index);
                }
            }
            if state == 1 {
                if d1 == 0 {
                    return None;
                } else {
                    let index: u8 = d1.leading_zeros() as u8;
                    d1 &= !(1u64 << (63 - index));
                    return Some(index + 64);
                }
            }
            None
        })
    }

    pub fn flip(&mut self) {
        *self ^= &BOARD_MASK;
    }

    pub fn bitand_batch(boards: &[BitBoard]) -> BitBoard {
        if boards.is_empty() {
            return BitBoard::from_u128(u128::MAX); // AND identity
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::bitand_batch_avx2(boards) };
            }
        }

        // Scalar fallback
        let mut result: BitBoard = boards[0];
        for board in &boards[1..] {
            result &= *board;
        }
        result
    }

    pub fn bitor_batch(boards: &[BitBoard]) -> BitBoard {
        if boards.is_empty() {
            return BitBoard::new(); // OR identity
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::bitor_batch_avx2(boards) };
            }
        }

        // Scalar fallback
        let mut result: BitBoard = boards[0];
        for board in &boards[1..] {
            result |= *board;
        }
        result
    }

    pub fn bitxor_batch(boards: &[BitBoard]) -> BitBoard {
        if boards.is_empty() {
            return BitBoard::new(); // XOR identity
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::bitxor_batch_avx2(boards) };
            }
        }

        // Scalar fallback
        let mut result: BitBoard = boards[0];
        for board in &boards[1..] {
            result ^= *board;
        }
        result
    }

    pub fn count_ones(&self) -> u32 {
        // The Rust compiler automatically optimizes this to `popcnt` on supported CPUs.
        self.data[0].count_ones() + self.data[1].count_ones()
    }

    pub fn shift_left_batch(boards: &[BitBoard], rhs: usize) -> Vec<BitBoard> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::shift_left_batch_avx2(boards, rhs) };
            }
        }
        // Scalar fallback
        boards.iter().map(|b| *b << rhs).collect()
    }

    pub fn shift_right_batch(boards: &[BitBoard], rhs: usize) -> Vec<BitBoard> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::shift_right_batch_avx2(boards, rhs) };
            }
        }
        // Scalar fallback
        boards.iter().map(|b| *b >> rhs).collect()
    }
}

const BOARD_MASK_U128: u128 = !((1u128 << (128 - LENGTH_OF_BOARD as u32)) - 1);
static BOARD_MASK: BitBoard = BitBoard {
    data: [(BOARD_MASK_U128 >> 64) as u64, BOARD_MASK_U128 as u64],
};

#[cfg(target_arch = "x86_64")]
/// Performs a bitwise XOR operation using SSE2 intrinsics.
///
/// # Safety
/// This function is unsafe because it uses SIMD intrinsics and assumes that the
/// `sse2` feature is available on the CPU.
unsafe fn sse2_xor(a: &BitBoard, b: &BitBoard) -> BitBoard {
    // Load the two 64-bit integers of each BitBoard into a 128-bit SSE register.
    let a_vec: __m128i = _mm_loadu_si128(a.data.as_ptr() as *const __m128i);
    let b_vec: __m128i = _mm_loadu_si128(b.data.as_ptr() as *const __m128i);

    // Perform the bitwise XOR operation on the 128-bit registers.
    let result: __m128i = _mm_xor_si128(a_vec, b_vec);

    // Store the result back into a BitBoard data array.
    let mut output: [u64; 2] = [0u64; 2];
    _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
    BitBoard { data: output }
}

#[cfg(target_arch = "x86_64")]
/// Performs a bitwise AND operation using SSE2 intrinsics.
///
/// # Safety
/// This function is unsafe because it uses SIMD intrinsics and assumes that the
/// `sse2` feature is available on the CPU.
unsafe fn sse2_bitand(a: &BitBoard, b: &BitBoard) -> BitBoard {
    // Load the two 64-bit integers of each BitBoard into a 128-bit SSE register.
    let a_vec: __m128i = _mm_loadu_si128(a.data.as_ptr() as *const __m128i);
    let b_vec: __m128i = _mm_loadu_si128(b.data.as_ptr() as *const __m128i);

    // Perform the bitwise AND operation on the 128-bit registers.
    let result: __m128i = _mm_and_si128(a_vec, b_vec);

    // Store the result back into a BitBoard data array.
    let mut output: [u64; 2] = [0u64; 2];
    _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
    BitBoard { data: output }
}

impl BitAnd for BitBoard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                return unsafe { sse2_bitand(&self, &rhs) };
            }
        }
        BitBoard {
            data: [self.data[0] & rhs.data[0], self.data[1] & rhs.data[1]],
        }
    }
}

impl BitAnd<&BitBoard> for &BitBoard {
    type Output = BitBoard;

    fn bitand(self, rhs: &BitBoard) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                return unsafe { sse2_bitand(self, rhs) };
            }
        }
        BitBoard {
            data: [self.data[0] & rhs.data[0], self.data[1] & rhs.data[1]],
        }
    }
}

impl BitAndAssign for BitBoard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.data[0] &= rhs.data[0];
        self.data[1] &= rhs.data[1];
    }
}

impl BitAndAssign<&BitBoard> for BitBoard {
    fn bitand_assign(&mut self, rhs: &BitBoard) {
        self.data[0] &= rhs.data[0];
        self.data[1] &= rhs.data[1];
    }
}

#[cfg(target_arch = "x86_64")]
/// Performs a bitwise OR operation using SSE2 intrinsics.
///
/// # Safety
/// This function is unsafe because it uses SIMD intrinsics and assumes that the
/// `sse2` feature is available on the CPU.
unsafe fn sse2_bitor(a: &BitBoard, b: &BitBoard) -> BitBoard {
    // Load the two 64-bit integers of each BitBoard into a 128-bit SSE register.
    let a_vec: __m128i = _mm_loadu_si128(a.data.as_ptr() as *const __m128i);
    let b_vec: __m128i = _mm_loadu_si128(b.data.as_ptr() as *const __m128i);

    // Perform the bitwise OR operation on the 128-bit registers.
    let result: __m128i = _mm_or_si128(a_vec, b_vec);

    // Store the result back into a BitBoard data array.
    let mut output: [u64; 2] = [0u64; 2];
    _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
    BitBoard { data: output }
}

impl BitOr for BitBoard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                return unsafe { sse2_bitor(&self, &rhs) };
            }
        }
        BitBoard {
            data: [self.data[0] | rhs.data[0], self.data[1] | rhs.data[1]],
        }
    }
}

impl BitOr<&BitBoard> for &BitBoard {
    type Output = BitBoard;

    fn bitor(self, rhs: &BitBoard) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                return unsafe { sse2_bitor(self, rhs) };
            }
        }
        BitBoard {
            data: [self.data[0] | rhs.data[0], self.data[1] | rhs.data[1]],
        }
    }
}

impl BitOrAssign for BitBoard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.data[0] |= rhs.data[0];
        self.data[1] |= rhs.data[1];
    }
}

impl BitOrAssign<&BitBoard> for BitBoard {
    fn bitor_assign(&mut self, rhs: &BitBoard) {
        self.data[0] |= rhs.data[0];
        self.data[1] |= rhs.data[1];
    }
}

impl BitXor for BitBoard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                return unsafe { sse2_xor(&self, &rhs) };
            }
        }
        BitBoard {
            data: [self.data[0] ^ rhs.data[0], self.data[1] ^ rhs.data[1]],
        }
    }
}

impl BitXor<&BitBoard> for &BitBoard {
    type Output = BitBoard;

    fn bitxor(self, rhs: &BitBoard) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                return unsafe { sse2_xor(self, rhs) };
            }
        }
        BitBoard {
            data: [self.data[0] ^ rhs.data[0], self.data[1] ^ rhs.data[1]],
        }
    }
}

impl BitXorAssign for BitBoard {
    fn bitxor_assign(&mut self, rhs: Self) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                *self = unsafe { sse2_xor(self, &rhs) };
                return;
            }
        }
        self.data[0] ^= rhs.data[0];
        self.data[1] ^= rhs.data[1];
    }
}

impl BitXorAssign<&BitBoard> for BitBoard {
    fn bitxor_assign(&mut self, rhs: &BitBoard) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                *self = unsafe { sse2_xor(self, rhs) };
                return;
            }
        }
        self.data[0] ^= rhs.data[0];
        self.data[1] ^= rhs.data[1];
    }
}

impl Shr<usize> for BitBoard {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        if rhs >= 128 {
            BitBoard { data: [0, 0] }
        } else if rhs >= 64 {
            BitBoard {
                data: [0, self.data[0] >> (rhs - 64)],
            }
        } else if rhs > 0 {
            BitBoard {
                data: [
                    self.data[0] >> rhs,
                    (self.data[1] >> rhs) | (self.data[0] << (64 - rhs)),
                ],
            }
        } else {
            self
        }
    }
}

impl ShrAssign<usize> for BitBoard {
    fn shr_assign(&mut self, rhs: usize) {
        *self = *self >> rhs;
    }
}

impl Shl<usize> for BitBoard {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        if rhs >= 128 {
            BitBoard { data: [0, 0] }
        } else if rhs >= 64 {
            BitBoard {
                data: [self.data[1] << (rhs - 64), 0],
            }
        } else if rhs > 0 {
            BitBoard {
                data: [
                    (self.data[0] << rhs) | (self.data[1] >> (64 - rhs)),
                    self.data[1] << rhs,
                ],
            }
        } else {
            self
        }
    }
}

impl ShlAssign<usize> for BitBoard {
    fn shl_assign(&mut self, rhs: usize) {
        *self = *self << rhs;
    }
}

pub fn generate_columns(column_nos: Vec<usize>) -> BitBoard {
    let mut bitboard: BitBoard = BitBoard::new();
    let mut first_row_mask: BitBoard = BitBoard::new();
    for column_no in column_nos {
        // Assumes column_no is 0-indexed (0-8) for the 9 playable columns
        // The first playable row starts at index 12 (1*11 + 1)
        let index: usize = 11 + (column_no + 1);
        first_row_mask |= BitBoard::from_u128(1u128 << (127 - index));
    }

    for _r in 0..LENGTH_OF_EDGE {
        bitboard |= first_row_mask;
        first_row_mask >>= LENGTH_OF_FRAME as usize;
    }
    bitboard
}

pub fn generate_column(column_no: usize) -> BitBoard {
    // Assumes column_no is 0-indexed (0-8) for the 9 playable columns
    let mut bitboard: BitBoard = BitBoard::new();
    let index: usize = 11 + (column_no + 1);
    let mut mask: BitBoard = BitBoard::from_u128(1u128 << (127 - index));
    for _r in 0..LENGTH_OF_EDGE {
        bitboard |= mask;
        mask >>= LENGTH_OF_FRAME as usize;
    }
    bitboard
}

// impl std::fmt::Display for BitBoard {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {

//         write!(f, "{}", convert_string(*self))
//     }
// }

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::BitBoard;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn shift_right_batch_avx2(boards: &[BitBoard], rhs: usize) -> Vec<BitBoard> {
        if rhs == 0 {
            return boards.to_vec();
        }
        if rhs >= 128 {
            return vec![BitBoard::new(); boards.len()];
        }

        let mut result: Vec<BitBoard> = Vec::with_capacity(boards.len());
        let mut i: usize = 0;

        // Process boards in chunks of 2 (256 bits)
        while i + 1 < boards.len() {
            let board_ptr: *const __m256i = boards.as_ptr().add(i) as *const __m256i;
            let v: __m256i = _mm256_loadu_si256(board_ptr);

            let shifted_v: __m256i = if rhs < 64 {
                let shift_vec: __m256i = _mm256_set1_epi64x(rhs as i64);
                let shifted_right: __m256i = _mm256_srlv_epi64(v, shift_vec);
                let shuffled_v: __m256i = _mm256_permute4x64_epi64(v, 0xA0); // [d0, d0, d2, d2]
                let shift_left_vec: __m256i = _mm256_set1_epi64x((64 - rhs) as i64);
                let shifted_left: __m256i = _mm256_sllv_epi64(shuffled_v, shift_left_vec);
                let mask: __m256i = _mm256_set_epi64x(-1, 0, -1, 0);
                let carry: __m256i = _mm256_and_si256(shifted_left, mask);
                _mm256_or_si256(shifted_right, carry)
            } else {
                let shift_vec: __m256i = _mm256_set1_epi64x((rhs - 64) as i64);
                let shuffled_v: __m256i = _mm256_permute4x64_epi64(v, 0xA0); // [d0, d0, d2, d2]
                let shifted: __m256i = _mm256_srlv_epi64(shuffled_v, shift_vec);
                let mask: __m256i = _mm256_set_epi64x(-1, 0, -1, 0);
                _mm256_and_si256(shifted, mask)
            };

            let mut output: [u64; 4] = [0u64; 4];
            _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, shifted_v);
            result.push(BitBoard {
                data: [output[0], output[1]],
            });
            result.push(BitBoard {
                data: [output[2], output[3]],
            });
            i += 2;
        }

        if i < boards.len() {
            result.push(boards[i] >> rhs);
        }

        result
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn shift_left_batch_avx2(boards: &[BitBoard], rhs: usize) -> Vec<BitBoard> {
        if rhs == 0 {
            return boards.to_vec();
        }
        if rhs >= 128 {
            return vec![BitBoard::new(); boards.len()];
        }

        let mut result: Vec<BitBoard> = Vec::with_capacity(boards.len());
        let mut i: usize = 0;

        while i + 1 < boards.len() {
            let board_ptr: *const __m256i = boards.as_ptr().add(i) as *const __m256i;
            let v: __m256i = _mm256_loadu_si256(board_ptr);

            let shifted_v: __m256i = if rhs < 64 {
                let shift_vec: __m256i = _mm256_set1_epi64x(rhs as i64);
                let shifted_left: __m256i = _mm256_sllv_epi64(v, shift_vec);
                let shuffled_v: __m256i = _mm256_permute4x64_epi64(v, 0xF5); // [d1, d1, d3, d3]
                let shift_right_vec: __m256i = _mm256_set1_epi64x((64 - rhs) as i64);
                let shifted_right: __m256i = _mm256_srlv_epi64(shuffled_v, shift_right_vec);
                let mask: __m256i = _mm256_set_epi64x(0, -1, 0, -1);
                let carry: __m256i = _mm256_and_si256(shifted_right, mask);
                _mm256_or_si256(shifted_left, carry)
            } else {
                let shift_vec: __m256i = _mm256_set1_epi64x((rhs - 64) as i64);
                let shuffled_v: __m256i = _mm256_permute4x64_epi64(v, 0xF5); // [d1, d1, d3, d3]
                let shifted: __m256i = _mm256_sllv_epi64(shuffled_v, shift_vec);
                let mask: __m256i = _mm256_set_epi64x(0, -1, 0, -1);
                _mm256_and_si256(shifted, mask)
            };

            let mut output: [u64; 4] = [0u64; 4];
            _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, shifted_v);
            result.push(BitBoard {
                data: [output[0], output[1]],
            });
            result.push(BitBoard {
                data: [output[2], output[3]],
            });
            i += 2;
        }

        if i < boards.len() {
            result.push(boards[i] << rhs);
        }

        result
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn bitand_batch_avx2(boards: &[BitBoard]) -> BitBoard {
        assert!(!boards.is_empty());

        if boards.len() == 1 {
            return boards[0];
        }

        // Process boards in chunks of 2.
        // acc_vec will hold two parallel accumulations.
        // Lane 1: boards[0] & boards[2] & ...
        // Lane 2: boards[1] & boards[3] & ...
        let mut acc_vec: __m256i = _mm256_loadu_si256(boards.as_ptr() as *const __m256i);

        let mut i: usize = 2;
        while i + 1 < boards.len() {
            let next_vec: __m256i = _mm256_loadu_si256(boards[i..].as_ptr() as *const __m256i);
            acc_vec = _mm256_and_si256(acc_vec, next_vec);
            i += 2;
        }

        // Reduce the two lanes in acc_vec.
        let mut temp_output: [u64; 4] = [0u64; 4];
        _mm256_storeu_si256(temp_output.as_mut_ptr() as *mut __m256i, acc_vec);

        let mut result: BitBoard = BitBoard {
            data: [temp_output[0], temp_output[1]],
        } & BitBoard {
            data: [temp_output[2], temp_output[3]],
        };

        // Handle the last board if the count is odd.
        if i < boards.len() {
            result &= boards[i];
        }

        result
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn bitor_batch_avx2(boards: &[BitBoard]) -> BitBoard {
        assert!(!boards.is_empty());

        if boards.len() == 1 {
            return boards[0];
        }

        let mut acc_vec: __m256i = _mm256_loadu_si256(boards.as_ptr() as *const __m256i);

        let mut i: usize = 2;
        while i + 1 < boards.len() {
            let next_vec: __m256i = _mm256_loadu_si256(boards[i..].as_ptr() as *const __m256i);
            acc_vec = _mm256_or_si256(acc_vec, next_vec);
            i += 2;
        }

        let mut temp_output: [u64; 4] = [0u64; 4];
        _mm256_storeu_si256(temp_output.as_mut_ptr() as *mut __m256i, acc_vec);

        let mut result: BitBoard = BitBoard {
            data: [temp_output[0], temp_output[1]],
        } | BitBoard {
            data: [temp_output[2], temp_output[3]],
        };

        if i < boards.len() {
            result |= boards[i];
        }

        result
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn bitxor_batch_avx2(boards: &[BitBoard]) -> BitBoard {
        assert!(!boards.is_empty());

        if boards.len() == 1 {
            return boards[0];
        }

        let mut acc_vec: __m256i = _mm256_loadu_si256(boards.as_ptr() as *const __m256i);

        let mut i: usize = 2;
        while i + 1 < boards.len() {
            let next_vec: __m256i = _mm256_loadu_si256(boards[i..].as_ptr() as *const __m256i);
            acc_vec = _mm256_xor_si256(acc_vec, next_vec);
            i += 2;
        }

        let mut temp_output: [u64; 4] = [0u64; 4];
        _mm256_storeu_si256(temp_output.as_mut_ptr() as *mut __m256i, acc_vec);

        let mut result: BitBoard = BitBoard {
            data: [temp_output[0], temp_output[1]],
        } ^ BitBoard {
            data: [temp_output[2], temp_output[3]],
        };

        if i < boards.len() {
            result ^= boards[i];
        }

        result
    }
}
