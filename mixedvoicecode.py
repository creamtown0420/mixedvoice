import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# =========================================================
# 1. パラメータ設定
# =========================================================
SAMPLE_RATE = 16000    # 音声読み込み時のサンプリングレート
DURATION = 2.0         # 解析する音声の長さ（秒）
N_FFT = 1024           # STFT で使う FFT サイズ
HOP_LENGTH = 512       # STFT のホップサイズ
N_MELS = 64            # メルスペクトログラムの周波数ビン数（任意）
TEST_SIZE = 0.2        # テストデータの割合

# =========================================================
# 2. データ読み込み用関数の定義
# =========================================================
def load_audio_file(file_path, duration=DURATION, sr=SAMPLE_RATE):
    """
    WAV ファイルを読み込み、先頭から duration 秒だけを返す。
    duration 秒に満たない場合はゼロ埋め（padding）する。
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    # duration秒より短い場合、パディング
    if len(audio) < int(sr * duration):
        pad_length = int(sr * duration) - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    return audio

def extract_features(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    ここでは例としてメルスペクトログラムを特徴量として抽出。
    STFT からスペクトログラムを作り、さらにメル尺度に変換。
    """
    # メルスペクトログラム (周波数ビン x フレーム数)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # 対数変換（対数メルスペクトログラム）
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # CNNに渡しやすい形状 (周波数ビン, フレーム数, チャネル数=1) を想定
    # MLPを使う場合は2次元を平坦化して1次元ベクトルにしても良い
    # ここでは MLP 前提として、2次元をベクトルにフラット化します
    feature_vector = log_mel_spec.flatten()
    
    return feature_vector

# =========================================================
# 3. データセット作成
# =========================================================
def create_dataset(dataset_dir="dataset"):
    """
    dataset_dir 以下に「mix_voice」「not_mix_voice」フォルダがあることを想定。
    それぞれの音声ファイルから特徴量を抽出し、ラベルとともに返す。
    """
    X = []
    y = []
    
    # クラスラベルを付与するための辞書
    class_labels = {
        "mix_voice": 1,
        "not_mix_voice": 0
    }
    
    for class_name, label in class_labels.items():
        folder_path = os.path.join(dataset_dir, class_name)
        # .wav ファイルを取得
        for file_path in glob.glob(folder_path + "/*.wav"):
            audio = load_audio_file(file_path)
            feature_vector = extract_features(audio)
            X.append(feature_vector)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    return X, y

# =========================================================
# 4. モデル構築
# =========================================================
def create_model(input_dim):
    """
    全結合層（MLP）のシンプルな例。入力次元 input_dim は、
    メルスペクトログラムを flatten したサイズ。
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')  # 2クラス分類 (0 or 1) に使う出力層
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# =========================================================
# 5. メイン処理：データ作成 → 学習 → 検証
# =========================================================
if __name__ == "__main__":
    # データセット作成
    X, y = create_dataset("dataset")
    print("X shape:", X.shape)  # (サンプル数, 特徴量次元)
    print("y shape:", y.shape)  # (サンプル数, )
    
    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, random_state=42)
    
    # モデルを構築
    input_dim = X_train.shape[1]
    model = create_model(input_dim)
    model.summary()
    
    # 学習
    epochs = 10
    batch_size = 16
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # テストデータで評価
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # =========================================================
    # 6. 推論のサンプル
    # =========================================================
    # 適当なファイルを読み込んで推論してみる
    test_file_path = "sample_input.wav"  # 例: 実際に判定したい音声ファイル
    if os.path.exists(test_file_path):
        audio_test = load_audio_file(test_file_path)
        feat_test = extract_features(audio_test)
        
        # 形状を (1, input_dim) にしてモデルに入力
        feat_test = np.expand_dims(feat_test, axis=0)
        pred = model.predict(feat_test)
        
        # シグモイド出力を閾値0.5 で二値化
        if pred[0][0] >= 0.5:
            print(f"{test_file_path} はミックスボイスの可能性が高いです。 (スコア={pred[0][0]:.4f})")
        else:
            print(f"{test_file_path} はミックスボイスではない可能性が高いです。 (スコア={pred[0][0]:.4f})")
    else:
        print(f"推論テスト用のファイルが見つかりません: {test_file_path}")
