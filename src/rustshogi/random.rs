use rand::{thread_rng, Rng, RngCore};

pub struct Random {
    pub rng: rand::prelude::ThreadRng,
    pub min: u16,
    pub max: u16,
}

impl Random {
    pub fn init() -> Self {
        Self {
            rng: thread_rng(),
            min: 0,
            max: 9,
        }
    }

    pub fn new(min: u16, max: u16) -> Self {
        Self {
            rng: thread_rng(),
            min,
            max,
        }
    }

    pub fn generate_one(&mut self) -> u16 {
        self.rng.gen_range(self.min..=self.max)
    }

    pub fn generate_multi(&mut self, length: u16) -> Vec<u16> {
        let mut result: Vec<u16> = Vec::new();
        for _i in 0..length {
            let r: u16 = self.rng.gen_range(self.min..=self.max);
            result.push(r);
        }
        result
    }

    // Fast scalar version using Lemire's method for unbiased range mapping

    pub fn generate_multi_fast(&mut self, length: u16) -> Vec<u16> {
        let mut result: Vec<u16> = Vec::with_capacity(length as usize);
        let range: u64 = u64::from(self.max - self.min + 1);
        if range <= 1 {
            return vec![self.min; length as usize];
        }

        let mut buffer: [u8; 4] = [0u8; 4];
        for _ in 0..length {
            self.rng.fill_bytes(&mut buffer);
            let rand_u32: u32 = u32::from_ne_bytes(buffer);
            let scaled: u16 = ((u64::from(rand_u32) * range) >> 32) as u16;
            result.push(scaled + self.min);
        }
        result
    }

    // SSE2 SIMD version

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    pub unsafe fn generate_multi_sse2(&mut self, length: u16) -> Vec<u16> {
        use std::arch::x86_64::*;

        let len: usize = length as usize;
        let mut result: Vec<u16> = Vec::with_capacity(len);

        let range: u32 = self.max as u32 - self.min as u32 + 1;
        if range <= 1 {
            return vec![self.min; len];
        }
        let min_val: u16 = self.min;

        let range_vec: __m128i = _mm_set1_epi32(range as i32);

        let mut i: usize = 0;
        let chunks: usize = len / 4; // 4 u32s at a time

        for _ in 0..chunks {
            let mut buffer: [u8; 16] = [0u8; 16];
            self.rng.fill_bytes(&mut buffer);

            let rands_vec: __m128i = _mm_loadu_si128(buffer.as_ptr() as *const __m128i);

            // Multiply low 2 u32s
            let products_low: __m128i = _mm_mul_epu32(rands_vec, range_vec);

            // Multiply high 2 u32s
            let rands_high: __m128i = _mm_srli_epi64(rands_vec, 32);
            let products_high: __m128i = _mm_mul_epu32(rands_high, range_vec);

            // Shift right
            let scaled_low: __m128i = _mm_srli_epi64(products_low, 32);
            let scaled_high: __m128i = _mm_srli_epi64(products_high, 32);

            // Pack u64 results to u32s. This gives [res_high_1, res_high_0, res_low_1, res_low_0]
            let scaled_u32: __m128i = _mm_packs_epi32(scaled_low, scaled_high);

            // Pack u32s to u16s.
            let scaled_u16: __m128i = _mm_packs_epi32(scaled_u32, _mm_setzero_si128());

            // Add min
            let min_vec: __m128i = _mm_set1_epi16(min_val as i16);
            let final_vec: __m128i = _mm_add_epi16(scaled_u16, min_vec);

            // Store
            let mut storage: [u16; 8] = [0u16; 8];
            _mm_storeu_si128(storage.as_mut_ptr() as *mut __m128i, final_vec);
            result.extend_from_slice(&storage[0..4]);

            i += 4;
        }

        // Handle remainder
        if i < len {
            for _ in i..len {
                result.push(self.generate_one());
            }
        }

        result
    }
}
