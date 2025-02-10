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
SAMPLE_RATE = 16000    # サンプリングレート
DURATION = 2.0         # 切り出す秒数
N_FFT = 1024           
HOP_LENGTH = 512       
N_MELS = 64            # メルスペクトログラムの周波数ビン数
TEST_SIZE = 0.2        

# =========================================================
# 2. データ読み込み & 特徴抽出 関数
# =========================================================
def load_audio_file(file_path, duration=DURATION, sr=SAMPLE_RATE):
    """
    音声ファイルを読み込み、先頭から duration 秒だけを切り出す。
    足りない場合はゼロ埋め。
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    if len(audio) < int(sr * duration):
        pad_length = int(sr * duration) - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    return audio

def extract_features(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    メルスペクトログラム (周波数ビン×フレーム数) を取り出し、
    値は対数変換したうえで (周波数ビン, フレーム数, 1) の形にする。
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 形状 (n_mels, time_frames)
    # この後CNNに入力するために、チャネル次元を追加
    # 最終形状: (n_mels, time_frames, 1)
    log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
    
    return log_mel_spec

# =========================================================
# 3. データセット作成
# =========================================================
def create_dataset(dataset_dir="dataset"):
    """
    dataset_dir 以下に 「mix_voice」 「not_mix_voice」 フォルダがあることを想定。
    """
    X = []
    y = []
    
    class_labels = {
        "mix_voice": 1,
        "not_mix_voice": 0
    }
    
    for class_name, label in class_labels.items():
        folder_path = os.path.join(dataset_dir, class_name)
        for file_path in glob.glob(folder_path + "/*.wav"):
            audio = load_audio_file(file_path)
            feat = extract_features(audio)
            X.append(feat)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    return X, y

# =========================================================
# 4. CNNモデルの定義
# =========================================================
def create_cnn_model(input_shape):
    """
    input_shape は (n_mels, time_frames, 1) を想定。
    CNN で 2D 畳み込みを行い、最後に Dense で二値分類。
    """
    model = keras.Sequential([
        # Conv2D(フィルタ数, カーネルサイズ, etc...)
        keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2,2)),
        
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        
        keras.layers.Flatten(),  # 2次元畳み込みの出力を1次元ベクトル化
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')  # 二値分類なので最終出力は1ユニットのシグモイド
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# =========================================================
# 5. メイン処理
# =========================================================
if __name__ == "__main__":
    # データセット読み込み
    X, y = create_dataset("dataset")
    print("X shape:", X.shape)  # (サンプル数, n_mels, time_frames, 1)
    print("y shape:", y.shape)  # (サンプル数,)
    
    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, random_state=42)
    
    # モデル作成
    input_shape = X_train.shape[1:]  # (n_mels, time_frames, 1)
    model = create_cnn_model(input_shape)
    model.summary()
    
    # 学習
    epochs = 10
    batch_size = 16
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # テストデータで評価
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # 推論サンプル
    test_file_path = "sample_input.wav"
    if os.path.exists(test_file_path):
        audio_test = load_audio_file(test_file_path)
        feat_test = extract_features(audio_test)
        # (n_mels, time_frames, 1) → (1, n_mels, time_frames, 1)
        feat_test = np.expand_dims(feat_test, axis=0)
        
        pred = model.predict(feat_test)
        if pred[0][0] >= 0.5:
            print(f"{test_file_path} はミックスボイスの可能性が高いです (スコア={pred[0][0]:.4f})")
        else:
            print(f"{test_file_path} はミックスボイスではない可能性が高いです (スコア={pred[0][0]:.4f})")
