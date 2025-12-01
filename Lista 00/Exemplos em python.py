import cv2
import numpy as np

# ===============================================================
# ETAPA 1: LEITURA DA IMAGEM E ENTRADA DOS PONTOS
# ===============================================================

# Lê imagem
imagem = cv2.imread("Lista 00/img_retificacao.png")
if imagem is None:
    raise FileNotFoundError("Não foi possível carregar a imagem. "
                            "Verifique o nome ou o caminho do arquivo.")

altura, largura, canais = imagem.shape
print(f"Imagem carregada: {largura}x{altura} com {canais} canais")

# Entrada manual dos 4 pontos (x, y)
print("\nDigite as coordenadas dos 4 pontos (em pixels):")
pontos = []
for i in range(4):
    x = float(input(f"  x{i+1}: "))
    y = float(input(f"  y{i+1}: "))
    pontos.append([x, y])
pontos = np.array(pontos, dtype=np.float32)

# ===============================================================
# ETAPA 2: RETIFICAÇÃO AFIM
# ===============================================================

# Calcula linhas e pontos de fuga (vanishing points)
l1 = np.cross(np.append(pontos[0], 1), np.append(pontos[1], 1))
l2 = np.cross(np.append(pontos[2], 1), np.append(pontos[3], 1))
m1 = np.cross(np.append(pontos[0], 1), np.append(pontos[2], 1))
m2 = np.cross(np.append(pontos[1], 1), np.append(pontos[3], 1))

# Pontos de fuga
v1 = np.cross(l1, l2)
v2 = np.cross(m1, m2)

# Linha do infinito (vanishing line)
vanish_line = np.cross(v1, v2)
vanish_line = vanish_line / vanish_line[2]

# Matriz de homografia para retificação afim
H1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [vanish_line[0], vanish_line[1], vanish_line[2]]
], dtype=np.float64)

# Aplica a retificação afim
img_affine = cv2.warpPerspective(imagem, H1, (largura, altura))
cv2.imwrite("affine.jpg", img_affine)
print("\n[OK] Retificação afim concluída -> 'affine.jpg' salva.")

# ===============================================================
# ETAPA 3: RETIFICAÇÃO MÉTRICA
# ===============================================================

# Coordenadas após aplicar H1
pontos_h = np.hstack((pontos, np.ones((4, 1))))
pontos_ret = (H1 @ pontos_h.T).T
pontos_ret /= pontos_ret[:, 2][:, None]

# Linhas nas coordenadas retificadas
l1 = np.cross(pontos_ret[0], pontos_ret[1])
l2 = np.cross(pontos_ret[2], pontos_ret[3])
m1 = np.cross(pontos_ret[0], pontos_ret[2])
m2 = np.cross(pontos_ret[1], pontos_ret[3])

# Monta o sistema Mx = b
M = np.array([
    [l1[0]*m1[0], l1[0]*m1[1] + l1[1]*m1[0]],
    [l2[0]*m2[0], l2[0]*m2[1] + l2[1]*m2[0]]
])
b = np.array([-l1[1]*m1[1], -l2[1]*m2[1]])

# Resolve para x (parâmetros da matriz S)
x = np.linalg.solve(M, b)

# Monta a matriz S
S = np.array([
    [x[0], x[1]],
    [x[1], 1]
])

# Decomposição SVD
U, D, Vt = np.linalg.svd(S)
A = U.T @ np.sqrt(np.diag(D)) @ Vt

# Matriz de homografia métrica
H2 = np.array([
    [A[0,0], A[0,1], 0],
    [A[1,0], A[1,1], 0],
    [0, 0, 1]
])

# Aplica transformação final
H2_inv = np.linalg.inv(H2)
img_scene = cv2.warpPerspective(img_affine, H2_inv, (largura, altura))
cv2.imwrite("scene.jpg", img_scene)
print("[OK] Retificação métrica concluída -> 'scene.jpg' salva.")

# ===============================================================
# ETAPA 4: EXIBIÇÃO FINAL
# ===============================================================

cv2.imshow("Original", imagem)
cv2.imshow("Afim", img_affine)
cv2.imshow("Métrica", img_scene)
cv2.waitKey(0)
cv2.destroyAllWindows()