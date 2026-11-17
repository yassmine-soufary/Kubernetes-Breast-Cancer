**Description courte pour GitHub**

Application de Machine Learning déployée sur Kubernetes avec Minikube pour classifier des tumeurs du sein et prédire leur surface moyenne.

**Présentation plus complète pour le README**

Ce projet déploie une stack de Machine Learning sur Kubernetes à partir du jeu de données Breast Cancer Wisconsin de scikit-learn. Il propose une classification des tumeurs en malignes ou bénignes et une régression pour prédire leur surface moyenne.

L’architecture comprend une API **FastAPI**, une interface **Streamlit**, un stockage **MinIO** pour les modèles et **Redis** pour le cache et les compteurs. Un Job Kubernetes entraîne les modèles, puis un Ingress expose l’application à l’adresse `http://bc.local`.

Le projet met en pratique les ConfigMaps, Secrets, sondes de santé, limites de ressources, autoscaling avec HPA, mises à jour progressives et rollback. Deux versions de modèles sont comparées, avec des algorithmes linéaires en v1 et Random Forest en v2.

Réalisé par **Yassmine Soufary** dans le cadre d’un TP Cloud et IA. Application à usage pédagogique.
