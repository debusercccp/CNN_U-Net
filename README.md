# CNN U-Net — Segmentazione di Immagini

Implementazione dell'architettura U-Net per la segmentazione semantica di immagini,
disponibile in due versioni: PyTorch e TensorFlow.
Il progetto include script per il training, l'inferenza e un ambiente Docker
per garantire riproducibilità indipendente dal sistema.

---
## Struttura
```
CNN_U-Net/
├── CNN_pytorch.py       # Implementazione U-Net in PyTorch
├── CNN_tensorflow.py    # Implementazione U-Net in TensorFlow
├── menu.sh              # Menu interattivo da terminale
├── requirements.txt     # Dipendenze Python
├── Dockerfile           # Immagine Docker con tutte le dipendenze
└── docker-compose.yml   # Configurazione dei servizi Docker
```
## Utilizzo

### Con Docker (consigliato)
docker-compose up

### Senza Docker
pip install -r requirements.txt
bash menu.sh

---

## Descrizione dei File

- **CNN_pytorch.py** — definizione del modello, training loop e inferenza in PyTorch
- **CNN_tensorflow.py** — stessa architettura reimplementata in TensorFlow
- **menu.sh** — interfaccia da terminale per avviare training, inferenza
  e altre operazioni senza dover ricordare i comandi specifici
- **Dockerfile** e **docker-compose.yml** — ambiente containerizzato
  per eliminare problemi di dipendenze e versioni

---

## Dipendenze

pip install -r requirements.txt
