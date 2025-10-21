 API de Reconhecimento Facial com FastAPI

Esta API permite **cadastrar e validar rostos de usuários** usando OpenCV e Dlib. Cada usuário é identificado por um vetor facial gerado pelo modelo `dlib_face_recognition_resnet_model_v1.dat`. Além disso, a API gera **tokens JWT** para usuários conhecidos.

---

##  Estrutura do Projeto

faceID.py # Arquivo principal da API
db.pkl # Banco de dados dos vetores faciais
shape_predictor_5_face_landmarks.dat
dlib_face_recognition_resnet_model_v1.dat

yaml
Copiar código

---

##  Requisitos

- Python 3.8+
- Pacotes Python:

```bash
pip install fastapi uvicorn opencv-python dlib numpy pyjwt
Modelos Dlib:

shape_predictor_5_face_landmarks.dat → detecta landmarks faciais

dlib_face_recognition_resnet_model_v1.dat → gera vetor facial

Configurações
PREDICTOR: caminho para o modelo de landmarks faciais

RECOG: caminho para o modelo de reconhecimento facial

DB_FILE: arquivo do banco de dados (pickle)

THRESH: limite de distância Euclidiana para considerar um rosto conhecido

SECRET_KEY: chave secreta para gerar JWT

Funções Principais
extract_vecs_from_image(img)
Extrai vetores faciais de uma imagem.

Entrada: imagem OpenCV (BGR)

Saída: lista de tuplas (vetor, rect)

save_db()
Salva o dicionário db no arquivo DB_FILE.

gerar_token(nome)
Gera um JWT válido por 1 hora para o usuário identificado pelo nome.

Endpoints
POST /cadastrar
Descrição: cadastra um novo usuário com foto

Parâmetros:

nome (string) → nome do usuário

foto (file) → imagem do rosto

Exemplo com curl:

bash
Copiar código
curl -X POST "http://127.0.0.1:8000/cadastrar" \
  -F "nome=Gustavo" \
  -F "foto=@/caminho/para/foto.jpg"
Resposta JSON:

json
Copiar código
{
  "msg": "Usuário Gustavo cadastrado com sucesso."
}
Erros Possíveis:

Nenhuma imagem válida

Nenhum rosto detectado

POST /validar
Descrição: valida a identidade de um rosto enviado

Parâmetros:

foto (file) → imagem do rosto

Exemplo com curl:

bash
Copiar código
curl -X POST "http://127.0.0.1:8000/validar" \
  -F "foto=@/caminho/para/foto.jpg"
Resposta JSON:

json
Copiar código
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
Erros Possíveis:

Nenhuma imagem válida

Nenhum rosto detectado

Execução
Rode a API localmente com:

bash
Copiar código
uvicorn faceID:app --reload
A documentação automática do FastAPI fica disponível em:

arduino
Copiar código
http://127.0.0.1:8000/docs

Observações
Não rode uvicorn.run dentro do script — use apenas o comando do terminal para evitar refresh infinito.

Banco de dados é persistente em db.pkl.

JWT é gerado apenas para usuários conhecidos.

Múltiplos rostos na mesma foto: apenas o maior rosto é usado no cadastro.