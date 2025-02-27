def inverse(e, phi):
  d = 0
  r = phi
  newt = 1
  newr = e

  while newr != 0:
    quotient = r // newr
    (d, newt) = (newt, d - quotient * newt)
    (r, newr) = (newr, r - quotient * newr)
  
  if r > 1:
    return None
  if d < 0:
    d += phi
  return d


def get_d(n, e):
  p = 0
  q = 0
  for p_i in range(2, n):
      if n % p_i == 0:
        p = p_i
        q = n // p_i
        break

  phi = (p - 1) * (q - 1)
  return phi, inverse(e, phi)


n = 471_090_785_117_207
e = 12377
c = 314999112281065205361706341517321987491098667

phi, d = get_d(n, e)
print("d:", d)
print("d * e % phi:", d * e % phi)
print("Расшифрованное сообщение:", pow(c, d, n))

output = open("output.txt", "w")

for k in range(0, n.bit_length()):
  output.write(f"k = {k}\n")
  # Определить количество бит в n
  n_bits = n.bit_length() - k

  # Представить C в двоичной системе
  C_binary = bin(c)[2:]  # Убираем префикс '0b'

  # Разбить C_binary на блоки по n_bits бит
  blocks = [int(C_binary[i:i+n_bits], 2) for i in range(0, len(C_binary), n_bits)]
  blocks = [pow(block, d, n) for block in blocks]
  blocks = [block.to_bytes(length=10) for block in blocks]

  # Вывести блоки
  for i, block in enumerate(blocks):
    output.write(f"Блок {i+1}: {block}\n")
  output.write("\n")

output.close()