# Tesi Pignedoli

Nei codici faccio riferimento a delle cartelle (/data, /model, /graph, /output, /pics) dove vengono salvati o caricati i file. In genere si possono scegliere i nomi dei file dalle prime righe di parametri. 
I vari parametri li ho impostati di volta in volta dal codice che poi salvavo e facevo ripartire.

## Preprocessing
il file pre_processing.py fa pulizia dei tweet, tokenization e stemming, in output un nuovo dataset con anche una colonna con il testo processato

## Tweet embedding
word2vec_model.py usa tutte le parole prese dai tweet pre_processati e costriuisce il modello
tweet_embedding.py usa il modello word2vec per fare emebdding delle parole e successivamente dei tweet

## Classification
classifier_confronto.py fa un confronto tra tre tipi di classificatori
calssification.py usa solo logistic regression e si può decidere se classficare i tweet embedding, i node embedding, entrambi e aplicare o meno un algoritmo di dimensionality reduction

## Network Analysis
nei file create_network.py, community_detection.py non ho seguito uno schema funzioni + esecuzione perchè non c'è bisogno di cambiare parametri
plot_community.py crea la visualizzazione del network
proximity.py e proximity_second_neigh.py creano l'embedding dei nodi (quindi degli utenti) con due modalità: confronto rispetto ai primi vicini, confronto rispetto ai primi e secondi vicini

ho integrato questi codici con un po di altre analisi fatte su diversi notebook, sopratutto per visualizzare i risultati.
