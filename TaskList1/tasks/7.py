def match(p, d):
    # Dict {index : letter}
    idxLetter = {idx: letter for idx, letter in enumerate(p) if letter != "*"}
    # True if all letters from created dict match string from data 
    return [letterData for letterData in d if all(letterData[i] == letterPattern for i, letterPattern in idxLetter.items())]

pattern = "a**a******"
data = ['aababacaa', 'cabaabcca', 'aaabbcbacb', 'acababbaab', 'abbacbcbcb']
print(match(pattern, data))