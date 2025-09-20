MINDVEST - Reconhecimento facial APP

Este projeto implementa um sistema simples de reconhecimento facial em Python, utilizando as bibliotecas dlib, OpenCV e numpy.

O sistema permite:

Reconhecimento em tempo real via webcam (opcional, se tiver câmera)

Cadastrar rostos por foto (sem precisar de webcam).

Validar rostos por foto, comparando com o banco de dados.

Os rostos cadastrados ficam armazenados em um arquivo db.pkl.

Estrutura do Projeto
Sprint-IOT/
│-- faceID.py        # Script principal
│-- db.pkl           # Banco de dados (gerado automaticamente)
│-- shape_predictor_5_face_landmarks.dat
│-- dlib_face_recognition_resnet_model_v1.dat
│-- README.md

Instalação

Instale as dependências:

pip install opencv-python dlib numpy


Observação: no Windows, a instalação do dlib pode ser mais difícil. Em caso de erro, instale via wheel pré-compilado:

Descubra sua versão do Python (ex.: 3.11, 3.12).

Baixe o .whl correspondente em: https://pypi.org/project/dlib/#files

Instale com:

pip install dlib-19.xx-cp311-cp311-win_amd64.whl

Modelos do Dlib

Baixe os arquivos de modelos e coloque-os na mesma pasta do script:

shape_predictor_5_face_landmarks.dat

dlib_face_recognition_resnet_model_v1.dat

Descompacte os .bz2 e use os arquivos .dat.

Como rodar

Execute o script:

python faceID.py


Você verá o menu:

=== MENU ===
1 - Cadastrar rosto por foto
2 - Validar rosto por foto
3 - Usar webcam (tempo real)
4 - Sair

Funcionalidades
1. Cadastrar rosto por foto

Escolha a opção 1.
Digite o nome da pessoa e o caminho da foto (exemplo: joao.jpg).

Exemplo:

Digite o nome da pessoa: Joao
Digite o caminho da foto: joao.jpg
Usuário cadastrado: Joao


O vetor facial é salvo no arquivo db.pkl.

2. Validar rosto por foto

Escolha a opção 2.
Digite o caminho da foto para validar.

Se o rosto já estiver cadastrado:

Resultado: Joao


Se não estiver:

Resultado: Desconhecido

3. Reconhecimento em tempo real (Webcam)

Escolha a opção 3.

Abre uma janela com a câmera.

Mostra o nome da pessoa reconhecida.

Pressione C para cadastrar uma nova pessoa direto da câmera.

Pressione Q para sair.

Observação: se não houver webcam, esta opção não funcionará.

4. Sair

Encerra o programa.

Banco de Dados

O arquivo db.pkl armazena os rostos cadastrados no formato pickle.
Cada chave é o nome da pessoa e cada valor é o vetor facial (128 floats).

Exemplo simplificado:

{
  "Joao": [0.123, -0.044, 0.532, ...],
  "Maria": [0.222, -0.137, 0.421, ...]
}

Observações

Se apagar o db.pkl, o banco será recriado vazio.