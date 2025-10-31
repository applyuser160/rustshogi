use rustshogi::color::{convert_from_string, convert_string, get_reverse_color, ColorType};

#[test]
fn test_color_from_u8() {
    let color = ColorType::from_u8(0);
    assert_eq!(color as u8, 0);
}

#[test]
fn test_color_to_bool() {
    let color = ColorType::from_u8(1);
    assert_eq!(color.to_bool(), true);
}

#[test]
fn test_color_get_reverse_color() {
    let color = get_reverse_color(ColorType::Black);
    assert_eq!(color as u8, ColorType::White as u8);
}

#[test]
fn test_color_convert_string() {
    let color = ColorType::from_u8(0);
    assert_eq!(convert_string(color), 'b');
}

#[test]
fn test_color_convert_from_string() {
    let color = convert_from_string('b');
    assert_eq!(color as u8, ColorType::Black as u8);
}
