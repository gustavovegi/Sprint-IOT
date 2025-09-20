import cv2, dlib, numpy as np, pickle, os

# Modelos do dlib
PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
THRESH = 0.6

# Carregar banco de dados
db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR)
rec = dlib.face_recognition_model_v1(RECOG)


def extract_vec(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = detector(rgb, 1)
    if len(rects) == 0:
        return None
    shape = sp(rgb, rects[0])
    chip = dlib.get_face_chip(rgb, shape)
    return np.array(rec.compute_face_descriptor(chip), dtype=np.float32)


def cadastrar(nome, foto_path):
    img = cv2.imread(foto_path)
    if img is None:
        print("Não consegui abrir a foto:", foto_path)
        return
    vec = extract_vec(img)
    if vec is None:
        print("Nenhum rosto encontrado na imagem.")
        return
    db[nome] = vec
    pickle.dump(db, open(DB_FILE, "wb"))
    print("Usuário cadastrado:", nome)


def validar(foto_path):
    img = cv2.imread(foto_path)
    if img is None:
        print("Não consegui abrir a foto:", foto_path)
        return
    vec = extract_vec(img)
    if vec is None:
        print("Nenhum rosto encontrado na imagem.")
        return
    nome, dist = "Desconhecido", 999
    for n, v in db.items():
        d = np.linalg.norm(vec - v)
        if d < dist:
            nome, dist = n, d
    if dist > THRESH:
        nome = "Desconhecido"
    print("Resultado:", nome)


def iniciar_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nenhuma câmera encontrada.")
        return

    print("Pressione 'c' para cadastrar, 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vec = extract_vec(frame)
        nome = "Nenhum rosto"
        if vec is not None:
            nome, dist = "Desconhecido", 999
            for n, v in db.items():
                d = np.linalg.norm(vec - v)
                if d < dist:
                    nome, dist = n, d
            if dist > THRESH:
                nome = "Desconhecido"

        cv2.putText(frame, nome, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255), 2)
        cv2.imshow("Reconhecimento Facial", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            user = input("Digite o nome para cadastro: ")
            cadastrar(user, frame)  # salva direto da câmera
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        print("\n=== MENU ===")
        print("1 - Cadastrar rosto por foto")
        print("2 - Validar rosto por foto")
        print("3 - Usar webcam (tempo real)")
        print("4 - Sair")

        op = input("Escolha uma opção: ")

        if op == "1":
            nome = input("Digite o nome da pessoa: ")
            foto = input("Digite o caminho da foto: ")
            cadastrar(nome, foto)

        elif op == "2":
            foto = input("Digite o caminho da foto para validar: ")
            validar(foto)

        elif op == "3":
            iniciar_camera()

        elif op == "4":
            print("Saindo...")
            break

        else:
            print("Opção inválida, tente novamente.")
