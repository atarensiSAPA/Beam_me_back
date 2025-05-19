
# Detecció d'Emocions Facials en Temps Real

Aquest projecte permet detectar cares humanes i reconèixer les seves emocions (com alegria, tristesa, cansament, etc.) a partir d’imatges o una càmera. Inicialment pensat per a l’ús en aules, també pot ser aplicat en entorns com empreses, esdeveniments o estudis de comportament.

## Funcionalitats

- Detecció automàtica de cares a partir d’imatges o càmera.
- Identificació de persones conegudes (si es proporciona una base prèvia).
- Classificació de l’emoció facial mitjançant un model IA.
- Enviament d’un acudit al professor si es detecten emocions negatives.
- Interfície web senzilla per carregar imatges o activar la càmera.

## Objectius

- Millorar l’ambient de l’aula mitjançant la detecció emocional.
- Automatitzar l’anàlisi emocional per fer la classe més dinàmica i empàtica.
- Crear una solució fàcil d’utilitzar i extensible a altres àmbits.

## Tecnologies utilitzades

- **Python**
- **Flask** (backend web)
- **Hugging Face Transformers** (model de classificació d’emocions)
- **face_recognition** (detecció i reconeixement facial)
- **OpenCV / PIL** (tractament d’imatges)
- **JavaScript / HTML / CSS** (frontend)
- **Azure Blob Storage** (per emmagatzematge en el núvol, opcional)

## 🚀 Instal·lació i execució

```bash
# Clona el repositori
git clone https://github.com/el_teu_usuari/deteccio-emocions.git
cd deteccio-emocions

# Crea un entorn virtual i activa'l
python -m venv venv
source venv/bin/activate  # o .\venv\Scripts\activate en Windows

# Instal·la les dependències
pip install -r requirements.txt

# Executa l’aplicació
python index.py
```

Obre el navegador a [http://localhost:5000](http://localhost:5000)

## Estructura del projecte

```
.
├── controller/
│   ├── face_recognition.py
│   ├── hugging_face.py
│   ├── validate_img.py
│   └── ...
├── images/
│   ├── known/
│   ├── unknown/
│   └── detected_faces/
├── templates/
│   └── index.html
├── index.py
└── requirements.txt
```

## Cas d’ús principal

- En un entorn educatiu, ajuda el professorat a identificar si la classe mostra emocions negatives i automatitza l’enviament d’un acudit per millorar l’ambient.

## Estat del projecte

- Funcionalitat principal implementada  
- Possibilitat d’integració amb vídeo en temps real i millora de precisió del model

## Crèdits

Aquest projecte ha estat desenvolupat com a projecte final del curs d’especialització en Intel·ligència Artificial i Big Data. Idea original suggerida per un professor. Gràcies!

## Llicència

Aquest projecte es distribueix sota una llicència [MIT](LICENSE) — és lliure d’utilitzar i modificar.
