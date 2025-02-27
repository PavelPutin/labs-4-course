import re


def tokenize(text):
  text = re.split(r'\s+|[,-.":]+', text, flags=re.MULTILINE)
  return [v for v in text if len(v) != 0]


RU_LETTER_FREQUENCIES = {
    'о': 0.1097, 'е': 0.0845, 'а': 0.0801, 'и': 0.0735, 'н': 0.0670,
    'т': 0.0626, 'с': 0.0547, 'р': 0.0473, 'в': 0.0454, 'л': 0.0440,
    'к': 0.0349, 'м': 0.0321, 'д': 0.0298, 'п': 0.0281, 'у': 0.0262,
    'я': 0.0201, 'ы': 0.0190, 'ь': 0.0174, 'г': 0.0170, 'з': 0.0165,
    'б': 0.0159, 'ч': 0.0144, 'й': 0.0121, 'х': 0.0097, 'ж': 0.0094,
    'ш': 0.0073, 'ю': 0.0064, 'ц': 0.0048, 'щ': 0.0036, 'э': 0.0032,
    'ф': 0.0026, 'ъ': 0.0004, 'ё': 0.0004
}
RU_LETTER_FREQUENCIES_LIST = [(key, value) for key, value in RU_LETTER_FREQUENCIES.items()]
RU_LETTER_FREQUENCIES_LIST.sort(key=lambda x: x[1], reverse=True)


EN_LETTER_FREQUENCIES = {
    'e': 0.1202, 't': 0.0910, 'a': 0.0812, 'o': 0.0768, 'i': 0.0731,
    'n': 0.0695, 's': 0.0628, 'r': 0.0602, 'h': 0.0592, 'd': 0.0432,
    'l': 0.0398, 'u': 0.0288, 'c': 0.0271, 'm': 0.0261, 'f': 0.0230,
    'y': 0.0211, 'w': 0.0209, 'g': 0.0203, 'p': 0.0182, 'b': 0.0149,
    'v': 0.0111, 'k': 0.0069, 'x': 0.0017, 'q': 0.0011, 'j': 0.0010, 'z': 0.0007
}

EN_LETTER_FREQUENCIES_LIST = [(key, value) for key, value in EN_LETTER_FREQUENCIES.items()]
EN_LETTER_FREQUENCIES_LIST.sort(key=lambda x: x[1], reverse=True)

def swap(mapping, a, b):
  mapping[a], mapping[b] = mapping[b], mapping[a]

def analyse(source, freq_list, prefix):
  text = ""
  with open(source, "r") as c1:
    text = c1.read()
  cleared_text = re.sub(r'\s|\n|[,-.":;/]', '', text, flags=re.MULTILINE|re.DOTALL)

  abc = [v[0] for v in freq_list]
  letter_frequencies = {l: 0 for l in abc}
  for letter in cleared_text:
    letter_frequencies[letter] += 1

  n = len(cleared_text)
  for key, value in letter_frequencies.items():
    letter_frequencies[key] = value / n

  letter_frequencies_list = [(key, value) for key, value in letter_frequencies.items()]
  letter_frequencies_list.sort(key=lambda x: x[1], reverse=True)

  letter_mapping = {}
  for v in zip(letter_frequencies_list, freq_list):
      letter_mapping[v[0][0]] = v[1][0]
      
  if prefix == "en_":
    swap(letter_mapping, "f", "l")
    swap(letter_mapping, "x", "q")
    swap(letter_mapping, "g", "b")
    swap(letter_mapping, "z", "m")
    swap(letter_mapping, "f", "p")
    swap(letter_mapping, "t", "x")
    swap(letter_mapping, "f", "b")
    swap(letter_mapping, "e", "j")
    swap(letter_mapping, "g", "b")
    swap(letter_mapping, "e", "b")
    swap(letter_mapping, "i", "d")
    marked = set(list("you but be consequence make copyright example when as is published if version program"))
  else:
    swap(letter_mapping, "й", "з")
    swap(letter_mapping, "щ", "ю")
    swap(letter_mapping, "э", "ъ")
    swap(letter_mapping, "е", "п")
    swap(letter_mapping, "я", "и")
    swap(letter_mapping, "ъ", "д")
    swap(letter_mapping, "б", "и")
    swap(letter_mapping, "к", "ф")
    swap(letter_mapping, "к", "ы")
    swap(letter_mapping, "ж", "б")
    swap(letter_mapping, "с", "ь")
    swap(letter_mapping, "б", "ь")
    swap(letter_mapping, "о", "а")
    swap(letter_mapping, "б", "а")
    swap(letter_mapping, "й", "а")
    swap(letter_mapping, "ы", "п")
    swap(letter_mapping, "н", "ч")
    swap(letter_mapping, "н", "ы")
    swap(letter_mapping, "л", "у")
    swap(letter_mapping, "ы", "а")
    swap(letter_mapping, "у", "а")
    swap(letter_mapping, "т", "ш")
    swap(letter_mapping, "т", "в")
    marked = set(list("во и а из подъезжали расфранченные стоящий клюве части разрешено торговцы этих другой прежние был ряд континенталь два на стороны когда-то охотный охотниками где одной праздниками головой"))

  with open(prefix + "frequency.txt", "w") as output:
    for v in zip(letter_frequencies_list, freq_list):
      mark = " +" if letter_mapping[v[0][0]] in marked else ""
      if letter_mapping[v[0][0]] in marked:
        print(letter_mapping[v[0][0]], end="")
      output.write(f"{v[0][0]} {v[0][1]:.5f} {v[1][0]} -> {letter_mapping[v[0][0]]} {v[1][1]:.5f} {abs(v[0][1] - v[1][1]):.5f}{mark}\n")
  print()
  with open(prefix + "map.txt", "w") as output:
    for k, v in letter_mapping.items():
      output.write(f"{k} -> {v}\n")

  m = ""
  for c in text:
    if c in letter_mapping.keys():
      m += letter_mapping[c]
    else:
      m += c

  with open(prefix + "output.txt", "w") as output:
    m = re.sub(r"\s{2,}", " ", m)
    m = re.sub(r"\n", " ", m)
    output.write(m)

  text = tokenize(text)
  m = tokenize(m)

  with open(prefix + "tokens.txt", "w") as output:
    for v in zip(text, m):
      output.write(f"{v[0]} -> {v[1]}\n")


analyse("lab5/c1.txt", RU_LETTER_FREQUENCIES_LIST, "ru_")
# analyse("lab5/c2.txt", EN_LETTER_FREQUENCIES_LIST, "en_")

