def format_filename(input_string: str) -> str:
    invalid_to_fullwidth: dict[str, str] = {
        '<': '＜',  # U+FF1C
        '>': '＞',  # U+FF1E
        ':': '：',  # U+FF1A
        '"': '＂',  # U+FF02
        '/': '／',  # U+FF0F
        '\\': '＼',  # U+FF3C
        '|': '｜',  # U+FF5C
        '?': '？',  # U+FF1F
        '*': '＊',  # U+FF0A
    }

    # Replace each invalid character with its fullwidth equivalent
    for char, fullwidth_char in invalid_to_fullwidth.items():
        input_string = input_string.replace(char, fullwidth_char)

    return input_string
