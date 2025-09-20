import cv2
import dlib
import numpy as np
import pickle
import os

# Modelos do dlib (coloque os .dat na mesma pasta)
PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
THRESH = 0.6  # limiar de distância para reconhecer

# Carregar/criar banco
db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR)
rec = dlib.face_recognition_model_v1(RECOG)


def extract_vecs_from_image(img):
    """
    Recebe BGR image (cv2), retorna lista de tuplas (vec, rect)
    rect é um dlib.rectangle
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = detector(rgb, 1)
    results = []
    for r in rects:
        shape = sp(rgb, r)
        chip = dlib.get_face_chip(rgb, shape)
        vec = np.array(rec.compute_face_descriptor(chip), dtype=np.float32)
        results.append((vec, r))
    return results


def cadastrar(nome, foto_path):
    img = cv2.imread(foto_path)
    if img is None:
        print("Não consegui abrir a foto:", foto_path)
        return
    vecs = extract_vecs_from_image(img)
    if not vecs:
        print("Nenhum rosto encontrado na imagem.")
        return

    # se tiver várias faces, pega a maior (maior área do rect)
    if len(vecs) > 1:
        areas = [( (r.right()-r.left()) * (r.bottom()-r.top()), i) for i, (_, r) in enumerate(vecs)]
        idx = max(areas)[1]
        vec = vecs[idx][0]
        print(f"Múltiplas faces encontradas. Cadastrando a face maior (índice {idx}).")
    else:
        vec = vecs[0][0]

    db[nome] = vec
    pickle.dump(db, open(DB_FILE, "wb"))
    print("Usuário cadastrado:", nome)


def validar_foto_e_mostrar(foto_path, janela=True):
    img = cv2.imread(foto_path)
    if img is None:
        print("Não consegui abrir a foto:", foto_path)
        return
    results = extract_vecs_from_image(img)
    if not results:
        print("Nenhum rosto encontrado na imagem.")
        return

    for vec, rect in results:
        nome, dist = "Desconhecido", 999
        for n, v in db.items():
            d = np.linalg.norm(vec - v)
            if d < dist:
                nome, dist = n, d
        if dist > THRESH:
            nome = "Desconhecido"

        # desenhar retângulo e etiqueta
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255), 2)
        label = f"{nome} ({dist:.2f})" if nome != "Desconhecido" else nome
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255), 2)
        print("Face:", label)

    if janela:
        cv2.imshow("Validação - Foto", img)
        print("Pressione qualquer tecla na janela para fechar.")
        cv2.waitKey(0)
        cv2.destroyWindow("Validação - Foto")


def iniciar_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nenhuma câmera encontrada.")
        return

    print("Webcam iniciada. Pressione 'c' para cadastrar a face maior, 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detectar todas as faces no frame
        res = extract_vecs_from_image(frame)
        # para cada face, tentar identificar
        for vec, rect in res:
            nome, dist = "Desconhecido", 999
            for n, v in db.items():
                d = np.linalg.norm(vec - v)
                if d < dist:
                    nome, dist = n, d
            if dist > THRESH:
                nome = "Desconhecido"

            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255), 2)
            label = f"{nome}" if nome != "Desconhecido" else nome
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255), 2)

        cv2.imshow("Reconhecimento Facial (webcam)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            # cadastrar a face maior do frame (se houver)
            if not res:
                print("Nenhuma face para cadastrar neste frame.")
                continue
            # escolher a face com maior área
            areas = [((r.right()-r.left()) * (r.bottom()-r.top()), i) for i, (_, r) in enumerate(res)]
            idx = max(areas)[1]
            vec, rect = res[idx]
            nome = input("Digite o nome para cadastrar a face selecionada: ")
            db[nome] = vec
            pickle.dump(db, open(DB_FILE, "wb"))
            print("Usuário cadastrado:", nome)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        print("1 - Cadastrar rosto por foto")
        print("2 - Validar rosto (foto com várias pessoas)")
        print("3 - Usar webcam (detecção de várias pessoas em tempo real)")
        print("4 - Sair")

        op = input("Escolha uma opção: ").strip()

        if op == "1":
            nome = input("Nome da pessoa: ").strip()
            foto = input("Caminho da foto: ").strip()
            cadastrar(nome, foto)

        elif op == "2":
            foto = input("Caminho da foto para validar: ").strip()
            validar_foto_e_mostrar(foto)

        elif op == "3":
            iniciar_camera()

        elif op == "4":
            print("Saindo")
            break

        else:
            print("Opção inválida. Tente novamente.")