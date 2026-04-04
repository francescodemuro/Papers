# Deep Learning for Pricing Complex Derivatives

This repository contains a research-style paper and supporting code on the application of deep learning methods to the pricing of complex financial derivatives.

## Overview

The project explores the intersection of:
- quantitative finance,
- stochastic calculus,
- partial differential equations,
- physics-informed neural networks,
- deep BSDE methods,
- stochastic volatility models such as Heston.

The goal is to study how deep neural networks can be used as scalable alternatives to traditional numerical pricing methods, especially in high-dimensional settings.

## Repository Structure

- `paper/`: LaTeX source of the paper and compiled PDF
- `src/`: Python implementation of neural pricing models
- `notebooks/`: experimental notebooks
- `results/`: numerical results, plots, and tables

## Main Topics

- Black-Scholes pricing PDE
- Physics-Informed Neural Networks (PINNs)
- Deep BSDE methods
- Basket options and high-dimensional pricing
- Heston stochastic volatility model
- Greeks computation via automatic differentiation
- No-arbitrage regularization

## How to Compile the Paper

Move into the `paper/` folder and run:

```bash
pdflatex main.tex
