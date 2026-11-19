# Synthèse des choix et résultats OPS

## Sujet et architecture

Le projet prédit l'annulation d'une réservation hôtelière à partir du jeu de données historique H1. Le modèle retenu est un arbre de décision. L'objectif est de démontrer une chaîne MLOps locale, reproductible et observable ; le score prédictif n'est pas un objectif imposé.

L'ingestion valide les données brutes et produit un découpage chronologique ainsi qu'un manifeste avec empreintes SHA-256. Prefect enchaîne la validation et l'entraînement, avec ingestion optionnelle si une source brute est fournie. Les données non conformes arrêtent le flux avant l'entraînement ; un nouvel essai exige la correction des données. La préparation et le modèle sont réunis dans un pipeline scikit-learn pour conserver le même traitement en apprentissage et en inférence.

MLflow conserve les paramètres, métriques, artefacts, empreintes de données et versions du modèle. Les seuils de rappel et de F1 servent de règle de promotion illustrative. L'API FastAPI résout l'alias `production` au démarrage, puis charge la version numérotée correspondante. SQLite en mode WAL conserve les prédictions, les feedbacks et les erreurs. Prometheus collecte les métriques ; Grafana présente deux tableaux de bord, pour l'API et pour le modèle/les traitements. Prefect signale les échecs du flux dans Telegram ; Alertmanager y transmet les alertes de disponibilité de l'API.

## Couverture et preuves

| Exigence | Mise en œuvre | Preuve ou état |
|---|---|---|
| Code modulaire et configuration | Modules `src/hotel_mlops/`, paramètres `config/project.toml`, dépendances séparées | 24 tests automatisés réussis dans l'environnement du projet |
| Installation reproductible | Procédure Python 3.11 et Docker Compose dans le README ; versions directes des dépendances fixées | Installation dans un environnement Python neuf et verrouillage complet non encore vérifiés |
| Ingestion et qualité | Validation des valeurs brutes, de la cible et des classes ; découpage par date ; manifeste et empreintes | Incident avec cible invalide démontré ; exécution de la nouvelle ingestion sur H1 encore à vérifier |
| Orchestration | Déploiement Prefect hebdomadaire, tâches de validation et d'entraînement, rapports persistants | Exécution planifiée déclenchée manuellement et terminée avec succès |
| Traçabilité et retour arrière | Versions MLflow, alias `production`, règle de promotion, commande de retour arrière | Changement d'alias, redémarrage de l'API et chargement de l'ancienne version démontrés |
| API et persistance | Validation des entrées, identifiant de requête, version du modèle, SQLite et feedbacks | Prédiction, feedback, rejet d'un doublon et santé de l'API vérifiés |
| Supervision et alertes | Métriques Prometheus, deux tableaux Grafana, règles d'alerte et Telegram | Échec de validation signalé par Prefect ; alertes API indisponible puis rétablie reçues |
| Charge et dimensionnement | Locust et échantillonnage des ressources Docker sur deux limites | Deux configurations mesurées avec le même scénario et la même image |

La dernière version `production` vérifiée par l'API est **v8**. Les métriques du modèle de référence sur le jeu de test sont : rappel **0,898**, F1 **0,740**, précision **0,629** et exactitude **0,776**. Ces résultats décrivent le jeu de test historique ; ils ne mesurent pas la qualité sur les futurs retours utilisateurs.

## Incident contrôlé et alertes

Une copie temporaire des données avec `IsCanceled = 2` a provoqué l'échec de la tâche Prefect de validation. L'entraînement a été bloqué et l'alias de production est resté inchangé. Le flux a enregistré l'échec et envoyé un message Telegram. Un arrêt temporaire de l'API a aussi déclenché `HotelApiUnavailable` dans Prometheus ; Alertmanager a envoyé les messages de déclenchement puis de résolution après le rétablissement du service.

Les prédictions et feedbacks restent associés à leur version de modèle. Les tableaux de bord affichent les retours par version ; les métriques de précision, rappel et F1 issues des feedbacks restent absentes tant qu'une version n'a pas reçu au moins 30 retours comportant les deux classes réelles. Ce seuil évite d'afficher un score sur un échantillon trop réduit, sans garantir l'absence de biais.

## Protocole et résultats de charge

Les deux essais ont été réalisés sur la même machine, avec la même image Docker (`sha256:f307466f6c31af91615352a103c658885a9addc3e42fbf385785df96ef9a081f`), le modèle **v8**, le même contenu de requête et la route `POST /predict`. Le scénario utilise **10 utilisateurs**, une montée à **2 utilisateurs/s**, une attente de **1 à 2 s** par utilisateur et une durée de **2 min**, montée en charge incluse. La santé de l'API a été vérifiée avant chaque essai pour exclure le chargement initial du modèle. Les essais ont été lancés successivement. La charge offerte, proche de 6,4 requêtes/s, ne cherche pas la saturation.

| Configuration | Limites API | Requêtes | Échecs | Débit | Moyenne | p95 | p99 | CPU max. échantillonné | Mémoire max. échantillonnée |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Petite | 0,5 CPU / 384 Mio | 764 | 0 | 6,42 req/s | 31,45 ms | 91 ms | 150 ms | 17,76 % | 189,7 Mio (49,4 % de la limite) |
| Standard | 1 CPU / 512 Mio | 769 | 0 | 6,48 req/s | 29,66 ms | 63 ms | 110 ms | 20,60 % | 187,2 Mio (36,6 % de la limite) |

Les critères fixés avant les essais étaient **zéro échec**, **p95 ≤ 200 ms** et **mémoire échantillonnée < 80 % de la limite**. Les deux configurations les respectent. La configuration standard réduit le p95 de 28 ms, mais la petite suffit pour cette charge. **Recommandation : 0,5 CPU et 384 Mio** pour la démonstration locale à environ 6,4 requêtes/s. Elle laisse 194,3 Mio entre le maximum de mémoire observé et la limite allouée, avec moitié moins de CPU et 128 Mio de mémoire alloués en moins que la configuration standard.

Les pourcentages CPU proviennent de Docker et sont relatifs à un cœur, pas à la limite CPU de chaque conteneur. L'échantillonnage peut manquer des pics brefs. Ces essais n'établissent ni le débit maximal, ni le comportement en cas de forte concurrence, ni le temps de démarrage à froid. Les limites sont à réévaluer si le trafic ou la taille du modèle augmente. Les fichiers détaillés des deux essais sont conservés localement sous `reports/load/`.

## Limites et vérifications restantes

- Les données sont historiques et rejouées ; il n'existe pas de flux hôtelier en direct. Le nouveau découpage par date peut différer des fichiers préparés antérieurs. Il faut vérifier l'ingestion sur H1 et conserver les empreintes et métriques de chaque exécution.
- Les feedbacks peuvent être sélectifs. Il n'y a pas de réentraînement automatique déclenché par les retours ou une dérive.
- L'API doit redémarrer après un changement d'alias MLflow. La disponibilité pendant le redémarrage, l'authentification et la promotion concurrente ne sont pas couvertes.
- Les notifications Telegram supposent que le poste, Docker et la connexion Internet fonctionnent. Alertmanager est une instance locale unique.
