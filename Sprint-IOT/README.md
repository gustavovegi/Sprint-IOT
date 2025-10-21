# Mindvest

permite cadastrar e validar rostos de usuários usando FastAPI, OpenCV e Dlib. Ela também gera tokens JWT para usuários conhecidos.

---

## Requisitos

```bash
pip install fastapi uvicorn opencv-python dlib numpy pyjwt

Certifique-se de ter os arquivos de modelos .dat no mesmo diretório:

shape_predictor_5_face_landmarks.dat

dlib_face_recognition_resnet_model_v1.dat

--------

- Rodando a API

uvicorn faceID:app --reload

A documentação interativa fica em:

http://127.0.0.1:8000/docs

Endpoints

POST /cadastrar
Descrição: Cadastra um usuário com foto

Parâmetros:
- nome (string)

- foto (file)

Exemplo de resposta: {
  "msg": "Usuário Gustavo cadastrado com sucesso."}


POST /validar

Descrição: Valida um rosto enviado

Parâmetros:
 - foto (file)

Exemplo de resposta:

{
  "faces": [
    {
      "nome": "Nome do usuário ou 'Desconhecido'",
      "distancia": 0.45,
      "bbox": [left, top, right, bottom],
      "token": "JWT token ou null"
    }
  ]
}

Não rode uvicorn.run dentro do script; use apenas o terminal.
O banco de dados é persistente em db.pkl.
JWT é gerado apenas para usuários conhecidos.
Se houver múltiplos rostos na imagem, apenas o maior é usado no cadastro.