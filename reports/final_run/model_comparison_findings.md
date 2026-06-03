# Model Comparison Findings

## Bot

- Best detection: Random Forest (0.958 recall, 24 test attacks).
- Most missed: Autoencoder + MLP (14 false negatives).

## Brute Force -Web

- Best detection: Random Forest (0.778 recall, 18 test attacks).
- Most missed: Logistic Regression (10 false negatives).

## Brute Force -XSS

- Best detection: Logistic Regression (1.000 recall, 20 test attacks).
- Most missed: Autoencoder + MLP (11 false negatives).

## DDOS attack-HOIC

- Best detection: Logistic Regression (1.000 recall, 618 test attacks).
- Most missed: Autoencoder + MLP (0 false negatives).

## DDOS attack-LOIC-UDP

- Best detection: Logistic Regression (1.000 recall, 15 test attacks).
- Most missed: Autoencoder + MLP (0 false negatives).

## DDoS attacks-LOIC-HTTP

- Best detection: Shallow MLP (1.000 recall, 20 test attacks).
- Most missed: Random Forest (1 false negatives).

## DoS attacks-GoldenEye

- Best detection: Logistic Regression (1.000 recall, 19 test attacks).
- Most missed: Deep MLP (1 false negatives).

## DoS attacks-Hulk

- Best detection: Logistic Regression (1.000 recall, 23 test attacks).
- Most missed: Random Forest (1 false negatives).

## DoS attacks-SlowHTTPTest

- Best detection: Logistic Regression (1.000 recall, 28 test attacks).
- Most missed: Autoencoder + MLP (0 false negatives).

## DoS attacks-Slowloris

- Best detection: Logistic Regression (1.000 recall, 23 test attacks).
- Most missed: Autoencoder + MLP (1 false negatives).

## FTP-BruteForce

- Best detection: Logistic Regression (1.000 recall, 18 test attacks).
- Most missed: Autoencoder + MLP (0 false negatives).

## Infilteration

- Best detection: Logistic Regression (0.299 recall, 1669 test attacks).
- Most missed: Random Forest (1335 false negatives).

## SQL Injection

- Best detection: Random Forest (1.000 recall, 20 test attacks).
- Most missed: Deep MLP (9 false negatives).

## SSH-Bruteforce

- Best detection: Logistic Regression (1.000 recall, 23 test attacks).
- Most missed: Autoencoder + MLP (0 false negatives).
