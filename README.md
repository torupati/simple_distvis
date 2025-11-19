# Simple Distribution Visualization

統計・機械学習の基本概念を可視化するStreamlitアプリケーションです。


## 概要

このプロジェクトは、統計学や機械学習の重要な概念を直感的に理解できるよう、インタラクティブな可視化ツールを提供します。

## 機能

- **Gaussian 1D Posterior**: ベイズ更新による事後分布の可視化
- **Sample Generator**: 様々な分布からのサンプル生成
- **K-Means Clustering**: K-Meansクラスタリングの可視化
- **Gaussian Mixture Model (GMM)**: EMアルゴリズムによるGMM学習
- **Markov Model**: マルコフモデルの可視化
- **Beta-Bayes**: ベータ分布を使ったベイズ推論
- **Diffusion Process**: 拡散過程の可視化
- **Normal Distribution 1D**: 正規分布の可視化
- **Bias-Variance Tradeoff**: バイアス・バリアンス分解の可視化

## インストール

### 前提条件

- Python 3.13以上
- [uv](https://docs.astral.sh/uv/)パッケージマネージャー

### セットアップ

1. リポジトリをクローン
```bash
git clone https://github.com/torupati/simple_distvis.git
cd simple_distvis
```

2. 依存関係をインストール
```bash
uv install
=======
gcloud auth login
gcloud projects create [YOUR_PROJECT_ID] --name="[PROJECT_NAME]"
gcloud config set project [YOUR_PROJECT_ID]
gcloud auth application-default set-quota-project [YOUR_PROJECT_ID]
gcloud config set run/region asia-northeast1
```

## 使い方

### アプリケーションの起動

```bash
uv run streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

### 各ページの説明

#### Gaussian 1D Posterior
- ガウス分布の事前分布と尤度関数から事後分布を計算
- ベイズ推論の基本概念を理解

#### Sample Generator
- 様々な確率分布からのサンプル生成
- 分布の性質とサンプリングの関係を学習

#### Markov Model
- マルコフ連鎖の状態遷移
- 定常分布の収束過程

#### Beta-Bayes
- ベータ分布を共役事前分布とするベイズ推論
- 逐次学習による分布の更新

#### Bias-Variance Tradeoff
- 機械学習におけるバイアス・バリアンス分解
- モデルの複雑さと性能の関係

## 開発

### テストの実行

```bash
uv run pytest
```

### カバレッジの確認

```bash
uv run pytest --cov=src
```

### 依存関係の管理

新しいパッケージを追加:
```bash
uv add <package_name>
```

開発依存関係を追加:
```bash
uv add --group dev <package_name>
```

## ライセンス

このプロジェクトは教育目的で作成されています。

## 貢献

バグ報告や機能提案は[Issues](https://github.com/torupati/simple_distvis/issues)でお願いします。