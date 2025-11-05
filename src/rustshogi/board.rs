use super::address::Address;
use super::bitboard::{
    generate_column, BitBoard, BIT_OF_FRAME, BIT_OF_LAST1_ZONE_BLACK, BIT_OF_LAST1_ZONE_WHITE,
    BIT_OF_LAST2_ZONE_BLACK, BIT_OF_LAST2_ZONE_WHITE, BIT_OF_PRO_ZONE_BLACK, BIT_OF_PRO_ZONE_WHITE,
    LENGTH_OF_EDGE, LENGTH_OF_FRAME,
};
use super::color::{get_reverse_color, ColorType};
use super::direction::{Direction, DirectionName};
use super::hand::Hand;
use super::move_pattern;
use super::moves::Move;
use super::pca::apply_pca_compression;
use super::piece::{MoveType, Piece, PieceType, PIECE_TYPE_NUMBER, PROMOTE};

use lru::LruCache;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use strum::IntoEnumIterator;

use std::hash::Hash;

const CACHE_SIZE: usize = 70000;

static MOVE_CACHE: Lazy<Mutex<LruCache<(Board, ColorType), Vec<Move>>>> =
    Lazy::new(|| Mutex::new(LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap())));

#[pyclass]
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Board {
    pub has_piece: BitBoard,
    pub player_prossesion: [BitBoard; ColorType::ColorNumber as usize],
    pub is_frame: BitBoard,
    pub able_pro: [BitBoard; ColorType::ColorNumber as usize],
    pub last_one: [BitBoard; ColorType::ColorNumber as usize],
    pub last_two: [BitBoard; ColorType::ColorNumber as usize],
    pub has_specific_piece: [BitBoard; PIECE_TYPE_NUMBER as usize],
    pub hand: Hand,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    fn is_a_has_specific_piece(&self, index: u8, piece_type: PieceType) -> bool {
        (self.has_specific_piece[piece_type as usize].to_u128() >> (127 - index)) & 1 != 0
    }

    fn drop(&mut self, index: u8) {
        let bit = 1u128 << (127 - index);
        let bit_mask = BitBoard::from_u128(bit);
        let not_bit_mask = BitBoard::from_u128(!bit);

        self.has_piece &= not_bit_mask;
        for player_prosession in self.player_prossesion.iter_mut() {
            *player_prosession &= not_bit_mask;
        }
        self.has_specific_piece[PieceType::None as usize] |= bit_mask;
        for has_specific_piece in self.has_specific_piece.iter_mut().skip(1) {
            *has_specific_piece &= not_bit_mask;
        }
    }

    fn move_standard(&mut self, from_index: u8, to_index: u8) {
        let piece_type = self.get_piece_type_from_index(from_index);
        let color_type = self.get_color_type_from_index(from_index);
        self.drop(from_index);
        self.deploy(to_index, piece_type, color_type);
    }

    fn move_to_hand(&mut self, index: u8, is_color_reverse: bool) {
        let mut piece_type = self.get_piece_type_from_index(index);
        if piece_type as u8 > PROMOTE {
            piece_type = PieceType::from_usize(piece_type as usize - PROMOTE as usize);
        }

        let mut color_type = self.get_color_type_from_index(index);
        if is_color_reverse {
            color_type = get_reverse_color(color_type);
        }

        self.drop(index);
        self.hand.add_piece(color_type, piece_type);
    }

    fn move_from_hand(&mut self, index: u8, piece_type: PieceType, color: ColorType) {
        self.hand.decrease_piece(color, piece_type);
        self.deploy(index, piece_type, color);
    }

    pub fn new() -> Self {
        let is_frame = BitBoard::from_u128(BIT_OF_FRAME);
        let mut has_specific_piece = BitBoard::from_u128(BIT_OF_FRAME);
        has_specific_piece.flip();
        Self {
            has_piece: BitBoard::new(),
            player_prossesion: [BitBoard::new(), BitBoard::new()],
            is_frame,
            able_pro: [
                BitBoard::from_u128(BIT_OF_PRO_ZONE_BLACK),
                BitBoard::from_u128(BIT_OF_PRO_ZONE_WHITE),
            ],
            last_one: [
                BitBoard::from_u128(BIT_OF_LAST1_ZONE_BLACK),
                BitBoard::from_u128(BIT_OF_LAST1_ZONE_WHITE),
            ],
            last_two: [
                BitBoard::from_u128(BIT_OF_LAST2_ZONE_BLACK),
                BitBoard::from_u128(BIT_OF_LAST2_ZONE_WHITE),
            ],
            has_specific_piece: [
                has_specific_piece,
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
                BitBoard::new(),
            ],
            hand: Hand::new(),
        }
    }

    pub fn deploy(&mut self, index: u8, piece_type: PieceType, color: ColorType) {
        let mask = 1u128 << (127 - index);
        let bit_mask = BitBoard::from_u128(mask);
        let not_bit_mask = BitBoard::from_u128(!mask);

        self.has_piece |= bit_mask;
        for (i, player_prossesion) in self.player_prossesion.iter_mut().enumerate() {
            if color == ColorType::from_u8(i as u8) {
                *player_prossesion |= bit_mask;
            } else {
                *player_prossesion &= not_bit_mask;
            }
        }
        for (i, has_specific_piece) in self.has_specific_piece.iter_mut().enumerate() {
            if piece_type == PieceType::from_usize(i) {
                *has_specific_piece |= bit_mask;
            } else {
                *has_specific_piece &= not_bit_mask;
            }
        }
    }

    pub fn startpos(&mut self) {
        // Initial position: (column, row, piece_type, color)
        let positions = [
            // Black pieces - 1st rank
            (1, 1, PieceType::Lance, ColorType::Black),
            (2, 1, PieceType::Knight, ColorType::Black),
            (3, 1, PieceType::Silver, ColorType::Black),
            (4, 1, PieceType::Gold, ColorType::Black),
            (5, 1, PieceType::King, ColorType::Black),
            (6, 1, PieceType::Gold, ColorType::Black),
            (7, 1, PieceType::Silver, ColorType::Black),
            (8, 1, PieceType::Knight, ColorType::Black),
            (9, 1, PieceType::Lance, ColorType::Black),
            // Black pieces - 2nd rank
            (2, 2, PieceType::Bishop, ColorType::Black),
            (8, 2, PieceType::Rook, ColorType::Black),
            // Black pieces - 3rd rank (pawns)
            (1, 3, PieceType::Pawn, ColorType::Black),
            (2, 3, PieceType::Pawn, ColorType::Black),
            (3, 3, PieceType::Pawn, ColorType::Black),
            (4, 3, PieceType::Pawn, ColorType::Black),
            (5, 3, PieceType::Pawn, ColorType::Black),
            (6, 3, PieceType::Pawn, ColorType::Black),
            (7, 3, PieceType::Pawn, ColorType::Black),
            (8, 3, PieceType::Pawn, ColorType::Black),
            (9, 3, PieceType::Pawn, ColorType::Black),
            // White pieces - 9th rank
            (1, 9, PieceType::Lance, ColorType::White),
            (2, 9, PieceType::Knight, ColorType::White),
            (3, 9, PieceType::Silver, ColorType::White),
            (4, 9, PieceType::Gold, ColorType::White),
            (5, 9, PieceType::King, ColorType::White),
            (6, 9, PieceType::Gold, ColorType::White),
            (7, 9, PieceType::Silver, ColorType::White),
            (8, 9, PieceType::Knight, ColorType::White),
            (9, 9, PieceType::Lance, ColorType::White),
            // White pieces - 8th rank
            (8, 8, PieceType::Bishop, ColorType::White),
            (2, 8, PieceType::Rook, ColorType::White),
            // White pieces - 7th rank (pawns)
            (1, 7, PieceType::Pawn, ColorType::White),
            (2, 7, PieceType::Pawn, ColorType::White),
            (3, 7, PieceType::Pawn, ColorType::White),
            (4, 7, PieceType::Pawn, ColorType::White),
            (5, 7, PieceType::Pawn, ColorType::White),
            (6, 7, PieceType::Pawn, ColorType::White),
            (7, 7, PieceType::Pawn, ColorType::White),
            (8, 7, PieceType::Pawn, ColorType::White),
            (9, 7, PieceType::Pawn, ColorType::White),
        ];

        for (column, row, piece_type, color) in positions.iter() {
            self.deploy(
                Address::from_numbers(*column, *row).to_index(),
                *piece_type,
                *color,
            );
        }

        self.hand = Hand::new();
    }

    pub fn input_board(&mut self, sfen: &str) {
        let startpos = String::from("startpos");
        if startpos == sfen {
            self.startpos();
            return;
        }

        let parts: Vec<&str> = sfen.split('/').collect();
        for (row, part) in parts.iter().enumerate().rev() {
            let mut column = 0;
            let mut chars = part.chars().peekable();

            while let Some(ch) = chars.next() {
                if ch.is_ascii_digit() {
                    let empty_spaces = ch.to_digit(10).unwrap() as usize;
                    column += empty_spaces;
                } else {
                    let piece_str = if ch == '+' {
                        // For promoted pieces, combine '+' with the next character
                        if let Some(next_ch) = chars.next() {
                            format!("+{}", next_ch)
                        } else {
                            // Treat as an error if there is no character after '+'
                            continue;
                        }
                    } else {
                        // For regular pieces
                        ch.to_string()
                    };

                    let piece = Piece::from_string(&piece_str);
                    let piece_type = piece.piece_type;
                    let owner = piece.owner;
                    let index =
                        Address::from_numbers((1 + column) as u8, (9 - row) as u8).to_index();
                    self.deploy(index, piece_type, owner);
                    column += 1;
                }
            }
        }
    }

    pub fn input_hand(&mut self, sfen: &str) {
        if sfen == "-" {
            return;
        }
        let mut current_sfen = sfen.chars();
        while let Some(ch) = current_sfen.next() {
            if ch.is_ascii_digit() {
                let consecutive = ch.to_digit(10).unwrap() as u8;
                let piece = Piece::from_char(current_sfen.next().unwrap());
                self.hand
                    .add_pieces(piece.owner, piece.piece_type, consecutive);
            } else {
                let piece = Piece::from_char(ch);
                self.hand.add_piece(piece.owner, piece.piece_type);
            }
        }
    }

    pub fn from_sfen(sfen: String) -> Self {
        let mut board = Self::new();
        let parts: Vec<&str> = sfen.split(" ").collect();
        board.input_board(parts[0]);
        if parts.len() > 1 {
            board.input_hand(parts[1]);
        }
        board
    }

    pub fn get_piece_type_from_index(&self, index: u8) -> PieceType {
        for piece_type in PieceType::iter() {
            if self.is_a_has_specific_piece(index, piece_type) {
                return piece_type;
            }
        }
        PieceType::None
    }

    pub fn get_color_type_from_index(&self, index: u8) -> ColorType {
        let has_a_piece = (self.has_piece.to_u128() >> (127 - index)) & 1 != 0;
        let is_black =
            (self.player_prossesion[ColorType::Black as usize].to_u128() >> (127 - index)) & 1 != 0;
        if !has_a_piece {
            ColorType::None
        } else if is_black {
            ColorType::Black
        } else {
            ColorType::White
        }
    }

    pub fn get_piece(&self, index: u8) -> Piece {
        let piece_type = self.get_piece_type_from_index(index);
        let color_type = self.get_color_type_from_index(index);

        Piece::from(color_type, piece_type)
    }

    pub fn to_string(&self) -> String {
        let mut result = String::new();

        // Board part
        for row in (1..=LENGTH_OF_EDGE).rev() {
            if row < 9 {
                result.push('/');
            }
            let mut empty_count = 0;
            for col in 1..=LENGTH_OF_EDGE {
                let index = Address::from_numbers(col, row).to_index();
                let piece = self.get_piece(index);
                if piece.piece_type != PieceType::None {
                    if empty_count > 0 {
                        result.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    let piece_char = piece.to_string();
                    result.push_str(&piece_char);
                } else {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                result.push_str(&empty_count.to_string());
            }
        }

        // Hand part
        result.push(' ');
        let mut hand_str = String::new();

        // Black's hand pieces
        for piece_type in [
            PieceType::King,
            PieceType::Gold,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Silver,
            PieceType::Knight,
            PieceType::Lance,
            PieceType::Pawn,
        ]
        .iter()
        {
            let count = self.hand.get_count(ColorType::Black, *piece_type);
            if count > 0 {
                if count > 1 {
                    hand_str.push_str(&count.to_string());
                }
                let piece_char = Piece::from(ColorType::Black, *piece_type).to_string();
                hand_str.push_str(&piece_char);
            }
        }

        // White's hand pieces
        for piece_type in [
            PieceType::King,
            PieceType::Gold,
            PieceType::Rook,
            PieceType::Bishop,
            PieceType::Silver,
            PieceType::Knight,
            PieceType::Lance,
            PieceType::Pawn,
        ]
        .iter()
        {
            let count = self.hand.get_count(ColorType::White, *piece_type);
            if count > 0 {
                if count > 1 {
                    hand_str.push_str(&count.to_string());
                }
                let piece_char = Piece::from(ColorType::White, *piece_type).to_string();
                hand_str.push_str(&piece_char);
            }
        }

        if hand_str.is_empty() {
            result.push('-');
        } else {
            result.push_str(&hand_str);
        }

        result
    }

    pub fn get_able_move_squares(&self, index: u8) -> BitBoard {
        let piece_type = self.get_piece_type_from_index(index);
        let color_type = self.get_color_type_from_index(index);
        let bit_board = BitBoard::from_u128(1u128 << (127 - index));
        let mut bit_movable = BitBoard::new();

        let move_patterns = match piece_type {
            PieceType::King => move_pattern::KING_MOVE_PATTERNS,
            PieceType::Gold => move_pattern::GOLD_MOVE_PATTERNS,
            PieceType::Rook => move_pattern::ROOK_MOVE_PATTERNS,
            PieceType::Bishop => move_pattern::BISHOP_MOVE_PATTERNS,
            PieceType::Silver => move_pattern::SILVER_MOVE_PATTERNS,
            PieceType::Knight => move_pattern::KNIGHT_MOVE_PATTERNS,
            PieceType::Lance => move_pattern::LANCE_MOVE_PATTERNS,
            PieceType::Pawn => move_pattern::PAWN_MOVE_PATTERNS,
            PieceType::Dragon => move_pattern::DRAGON_MOVE_PATTERNS,
            PieceType::Horse => move_pattern::HORSE_MOVE_PATTERNS,
            PieceType::ProSilver => move_pattern::PRO_SILVER_MOVE_PATTERNS,
            PieceType::ProKnight => move_pattern::PRO_KNIGHT_MOVE_PATTERNS,
            PieceType::ProLance => move_pattern::PRO_LANCE_MOVE_PATTERNS,
            PieceType::ProPawn => move_pattern::PRO_PAWN_MOVE_PATTERNS,
            _ => &[],
        };

        for pattern in move_patterns {
            let mut direction = Direction::new(pattern.direction);
            if color_type == ColorType::White {
                direction.reverse();
            }

            match pattern.move_type {
                MoveType::Short => {
                    let shift_number = LENGTH_OF_FRAME as i8 * direction.vertical_vector
                        + direction.horizon_vector;
                    let target_board = if shift_number > 0 {
                        bit_board << shift_number as usize
                    } else {
                        bit_board >> shift_number.unsigned_abs() as usize
                    };

                    if (self.is_frame & target_board) == BitBoard::new()
                        && (self.player_prossesion[color_type as usize] & target_board)
                            == BitBoard::new()
                    {
                        bit_movable |= target_board;
                    }
                }
                MoveType::Long => {
                    for i in 1..LENGTH_OF_EDGE {
                        let shift_number = (LENGTH_OF_FRAME as i8 * direction.vertical_vector
                            + direction.horizon_vector)
                            * i as i8;
                        let target_board = if shift_number > 0 {
                            bit_board << shift_number as usize
                        } else {
                            bit_board >> shift_number.unsigned_abs() as usize
                        };

                        if (self.is_frame & target_board) != BitBoard::new() {
                            break;
                        }
                        if (self.player_prossesion[color_type as usize] & target_board)
                            != BitBoard::new()
                        {
                            break;
                        }

                        bit_movable |= target_board;

                        if (self.player_prossesion[get_reverse_color(color_type) as usize]
                            & target_board)
                            != BitBoard::new()
                        {
                            break;
                        }
                    }
                }
                MoveType::Hop => {
                    let up_direction = if color_type == ColorType::White {
                        Direction::new(DirectionName::Down)
                    } else {
                        Direction::new(DirectionName::Up)
                    };
                    let v = direction.vertical_vector + up_direction.vertical_vector;
                    let h = direction.horizon_vector;
                    let shift_number = LENGTH_OF_FRAME as i8 * v + h;
                    let target_board = if shift_number > 0 {
                        bit_board << shift_number as usize
                    } else {
                        bit_board >> shift_number.unsigned_abs() as usize
                    };

                    if (self.is_frame & target_board) == BitBoard::new()
                        && (self.player_prossesion[color_type as usize] & target_board)
                            == BitBoard::new()
                    {
                        bit_movable |= target_board;
                    }
                }
                MoveType::None => {}
            }
        }

        bit_movable
    }

    pub fn get_able_pro_move_squares(&self, index: u8, bit_movable: BitBoard) -> BitBoard {
        let result = BitBoard::new();
        let piece_type = self.get_piece_type_from_index(index);
        let color_type = self.get_color_type_from_index(index);
        let bit_board = BitBoard::from_u128(1u128 << (127 - index));
        let is_able_pro = Piece::able_pro(piece_type);
        if !is_able_pro {
            return result;
        }

        let pro_area = self.able_pro[color_type as usize];

        if (bit_board & pro_area) != BitBoard::new() {
            return bit_movable;
        }

        bit_movable & pro_area
    }

    pub fn get_able_drop_squares(&self, color: ColorType, piece_type: PieceType) -> BitBoard {
        let none = self.has_specific_piece[PieceType::None as usize];
        let mut last_not_two = self.last_two[color as usize];
        let mut last_not_one = self.last_one[color as usize];
        last_not_two.flip();
        last_not_one.flip();

        match piece_type {
            PieceType::Gold => none,
            PieceType::Rook => none,
            PieceType::Bishop => none,
            PieceType::Silver => none,
            PieceType::Knight => none & last_not_two,
            PieceType::Lance => none & last_not_one,
            PieceType::Pawn => {
                let pawn = self.has_specific_piece[PieceType::Pawn as usize]
                    & self.player_prossesion[color as usize];
                let mut double_pawn = BitBoard::new();

                for index in pawn.get_trues_iter() {
                    double_pawn |=
                        generate_column((Address::from_number(index).get_column() - 1) as usize);
                }

                let mut not_double_pawn = double_pawn;
                not_double_pawn.flip();

                none & (last_not_one & not_double_pawn)
            }
            _ => BitBoard::new(),
        }
    }

    pub fn search_moves(&self, color: ColorType, use_cache: bool) -> Vec<Move> {
        if use_cache {
            let mut cache = MOVE_CACHE.lock().unwrap();
            if let Some(moves) = cache.get(&(self.clone(), color)) {
                return moves.clone();
            }
        }

        let mut vector_move: Vec<Move> = Vec::with_capacity(128);

        let player_board = if color.to_bool() {
            &self.player_prossesion[ColorType::White as usize]
        } else {
            &self.player_prossesion[ColorType::Black as usize]
        };

        for player_board_index in player_board.get_trues_iter() {
            let move_board = self.get_able_move_squares(player_board_index);
            vector_move.extend(move_board.get_trues_iter().map(|move_index| {
                Move::from_standart(
                    Address::from_number(player_board_index),
                    Address::from_number(move_index),
                    false,
                )
            }));

            let pro_board = self.get_able_pro_move_squares(player_board_index, move_board);
            vector_move.extend(pro_board.get_trues_iter().map(|move_index| {
                Move::from_standart(
                    Address::from_number(player_board_index),
                    Address::from_number(move_index),
                    true,
                )
            }));
        }

        let player_hand_pieces = self.hand.get_player_pieces(color);
        for player_hand_piece in player_hand_pieces {
            let move_board =
                self.get_able_drop_squares(player_hand_piece.owner, player_hand_piece.piece_type);
            vector_move.extend(move_board.get_trues_iter().map(|move_index| {
                Move::from_drop(player_hand_piece, Address::from_number(move_index))
            }));
        }

        // Save to cache
        if use_cache {
            let mut cache = MOVE_CACHE.lock().unwrap();
            cache.put((self.clone(), color), vector_move.clone());
        }

        vector_move
    }

    pub fn execute_move(&mut self, moves: &Move) {
        let is_drop = moves.get_is_drop();
        let to_index = moves.get_to().to_index();
        let mut piece = Piece::new();
        let mut from_index: u8 = 0;

        if is_drop {
            piece = moves.get_piece();
        } else {
            from_index = moves.get_from().to_index();
        }

        if (self.has_piece.to_u128() >> (127 - to_index)) & 1 != 0 {
            self.move_to_hand(to_index, true);
        }

        if is_drop {
            self.move_from_hand(to_index, piece.piece_type, piece.owner);
        } else {
            // If it's a promotion move, change the piece to its promoted version
            if moves.get_is_promote() {
                let piece_type = self.get_piece_type_from_index(from_index);
                let color_type = self.get_color_type_from_index(from_index);
                self.drop(from_index);

                // Change piece to promoted version
                let promoted_piece_type = match piece_type {
                    PieceType::Rook => PieceType::Dragon,
                    PieceType::Bishop => PieceType::Horse,
                    PieceType::Silver => PieceType::ProSilver,
                    PieceType::Knight => PieceType::ProKnight,
                    PieceType::Lance => PieceType::ProLance,
                    PieceType::Pawn => PieceType::ProPawn,
                    _ => piece_type, // If the piece cannot be promoted, it remains the same
                };

                self.deploy(to_index, promoted_piece_type, color_type);
            } else {
                self.move_standard(from_index, to_index);
            }
        }
    }

    pub fn is_finished(&self) -> (bool, ColorType) {
        let winner;
        let is_finish = self.has_specific_piece[PieceType::King as usize]
            .to_u128()
            .count_ones()
            != ColorType::ColorNumber as u32;
        if is_finish {
            let is_black_win = (self.has_specific_piece[PieceType::King as usize]
                & self.player_prossesion[ColorType::Black as usize])
                != BitBoard::new();
            if is_black_win {
                winner = ColorType::Black;
            } else {
                winner = ColorType::White;
            }
        } else {
            winner = ColorType::None;
        }
        (is_finish, winner)
    }

    pub fn to_vector(&self, target_dimensions: Option<usize>) -> Vec<f32> {
        // Get all features
        let mut features = Vec::with_capacity(2320);

        // Basic board information (2304 dimensions) - Sequentially arrange all BitBoard values for each square
        let mut board_vector = [0f32; 2304];

        let bitboards = [
            self.has_piece,
            self.player_prossesion[ColorType::White as usize],
            self.player_prossesion[ColorType::Black as usize],
            self.has_specific_piece[PieceType::None as usize],
            self.has_specific_piece[PieceType::King as usize],
            self.has_specific_piece[PieceType::Gold as usize],
            self.has_specific_piece[PieceType::Rook as usize],
            self.has_specific_piece[PieceType::Bishop as usize],
            self.has_specific_piece[PieceType::Silver as usize],
            self.has_specific_piece[PieceType::Knight as usize],
            self.has_specific_piece[PieceType::Lance as usize],
            self.has_specific_piece[PieceType::Pawn as usize],
            self.has_specific_piece[PieceType::Dragon as usize],
            self.has_specific_piece[PieceType::Horse as usize],
            self.has_specific_piece[PieceType::ProSilver as usize],
            self.has_specific_piece[PieceType::ProKnight as usize],
            self.has_specific_piece[PieceType::ProLance as usize],
            self.has_specific_piece[PieceType::ProPawn as usize],
        ];

        // For each square (index), sequentially arrange all BitBoard values related to that square
        for index in 0..128 {
            let mut bitboard_offset = 0;
            for bitboard in &bitboards {
                let u128_value = bitboard.to_u128();
                board_vector[index * 18 + bitboard_offset] =
                    if (u128_value >> (127 - index)) & 1 != 0 {
                        1.0
                    } else {
                        0.0
                    };
                bitboard_offset += 1;
            }
        }

        features.extend_from_slice(&board_vector);

        // Hand piece information (16 dimensions: 2 colors × 8 piece types)
        let hand_features = self.hand.to_vector();
        features.extend_from_slice(&hand_features);

        // Total: 2304 + 16 = 2320 dimensions

        // If dimensionality reduction is needed
        if let Some(target_dims) = target_dimensions {
            if target_dims < features.len() {
                // Dimensionality reduction by PCA
                return apply_pca_compression(&features, target_dims);
            }
        }

        features
    }
}

#[pymethods]
impl Board {
    #[new]
    #[pyo3(signature = (sfen = "startpos".to_string()))]
    pub fn new_for_python(sfen: String) -> Self {
        Self::from_sfen(sfen)
    }

    pub fn __repr__(&self) -> String {
        format!("Board(sfen={})", self.to_string())
    }

    pub fn __str__(&self) -> String {
        format!("Board(sfen={})", self.to_string())
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.has_piece == other.has_piece
            && self.player_prossesion == other.player_prossesion
            && self.is_frame == other.is_frame
            && self.able_pro == other.able_pro
            && self.last_one == other.last_one
            && self.last_two == other.last_two
            && self.has_specific_piece == other.has_specific_piece
            && self.hand == other.hand
    }

    pub fn __ne__(&self, other: &Self) -> bool {
        self.has_piece != other.has_piece
            || self.player_prossesion != other.player_prossesion
            || self.is_frame != other.is_frame
            || self.able_pro != other.able_pro
            || self.last_one != other.last_one
            || self.last_two != other.last_two
            || self.has_specific_piece != other.has_specific_piece
            || self.hand != other.hand
    }

    #[pyo3(name = "deploy")]
    pub fn python_deploy(&mut self, address: &Address, piece_type: PieceType, color: ColorType) {
        self.deploy(address.to_index(), piece_type, color);
    }

    #[pyo3(name = "startpos")]
    pub fn python_startpos(&mut self) {
        self.startpos();
    }

    #[pyo3(name = "get_piece")]
    pub fn python_get_piece(&self, address: &Address) -> Piece {
        self.get_piece(address.to_index())
    }

    #[pyo3(name = "search_moves")]
    pub fn python_search_moves(&self, color: ColorType) -> Vec<Move> {
        self.search_moves(color, true)
    }

    #[pyo3(name = "execute_move")]
    pub fn python_execute_move(&mut self, moves: &Move) {
        self.execute_move(moves);
    }

    #[pyo3(name = "is_finished")]
    pub fn python_is_finished(&self) -> (bool, ColorType) {
        self.is_finished()
    }

    #[pyo3(name = "to_vector")]
    #[pyo3(signature = (target_dimensions = None))]
    pub fn python_to_vector(&self, target_dimensions: Option<usize>) -> Vec<f32> {
        self.to_vector(target_dimensions)
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}
