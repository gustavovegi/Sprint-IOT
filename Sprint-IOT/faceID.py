from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import dlib
import numpy as np
import pickle
import os
import tempfile
import jwt
import datetime

# === Configurações ===
PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
THRESH = 0.6
SECRET_KEY = "minha_chave_secreta"  # para gerar o token JWT

db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR)
rec = dlib.face_recognition_model_v1(RECOG)

app = FastAPI(title="API de Reconhecimento Facial")

# === Funções internas ===
def extract_vecs_from_image(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = detector(rgb, 1)
    results = []
    for r in rects:
        shape = sp(rgb, r)
        chip = dlib.get_face_chip(rgb, shape)
        vec = np.array(rec.compute_face_descriptor(chip), dtype=np.float32)
        results.append((vec, r))
    return results

def save_db():
    pickle.dump(db, open(DB_FILE, "wb"))

def gerar_token(nome):
    payload = {
        "sub": nome,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

# === Endpoints ===
@app.post("/cadastrar")
async def cadastrar(nome: str = Form(...), foto: UploadFile = File(...)):
    suffix = os.path.splitext(foto.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await foto.read())
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)
    if img is None:
        return JSONResponse({"erro": "Não consegui abrir a foto."}, status_code=400)

    vecs = extract_vecs_from_image(img)
    if not vecs:
        return JSONResponse({"erro": "Nenhum rosto encontrado."}, status_code=400)

    if len(vecs) > 1:
        areas = [((r.right()-r.left())*(r.bottom()-r.top()), i) for i, (_, r) in enumerate(vecs)]
        idx = max(areas)[1]
        vec = vecs[idx][0]
    else:
        vec = vecs[0][0]

    db[nome] = vec
    save_db()
    os.unlink(tmp_path)
    return {"msg": f"Usuário {nome} cadastrado com sucesso."}

@app.post("/validar")
async def validar(foto: UploadFile = File(...)):
    suffix = os.path.splitext(foto.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await foto.read())
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)
    if img is None:
        return JSONResponse({"erro": "Não consegui abrir a foto."}, status_code=400)

    results = extract_vecs_from_image(img)
    if not results:
        return JSONResponse({"erro": "Nenhum rosto encontrado."}, status_code=400)

    faces = []
    for vec, rect in results:
        nome, dist = "Desconhecido", 999
        for n, v in db.items():
            d = np.linalg.norm(vec - v)
            if d < dist:
                nome, dist = n, d
        if dist > THRESH:
            nome = "Desconhecido"

        # gerar token apenas se usuário for conhecido
        token = gerar_token(nome) if nome != "Desconhecido" else None

        faces.append({
            "nome": nome,
            "distancia": float(dist),
            "bbox": [rect.left(), rect.top(), rect.right(), rect.bottom()],
            "token": token
        })

    os.unlink(tmp_path)
    return {"faces": faces}

if __name__ == "__main__":
    print("Rodando")