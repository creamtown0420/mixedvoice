import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. パラメータ設定
# =========================================================
SAMPLE_RATE = 16000     # サンプリングレート
DURATION = 2.0          # 切り出す秒数
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
FMIN = 80               # 人の声を想定した最小周波数 (Hz)
FMAX = 8000             # 最大周波数 (Hz) - sr=16000ならナイキスト周波数が8000Hz
TEST_SIZE = 0.2         # テストデータ割合
RANDOM_STATE = 42       # 再現性のための乱数シード

# データ拡張（必要に応じて True/False を切り替え）
ENABLE_DATA_AUGMENT = False  # ピッチシフトやタイムストレッチを行う場合は True

# =========================================================
# 2. データ読み込み＆前処理関数
# =========================================================
def load_audio_file(file_path, duration=DURATION, sr=SAMPLE_RATE):
    """
    WAV ファイルを読み込み、先頭から duration 秒だけを返す。
    短い場合はゼロ埋め（パディング）する。
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    if len(audio) < int(sr * duration):
        pad_length = int(sr * duration) - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    return audio

def apply_data_augmentation(audio, sr):
    """
    データ拡張（Data Augmentation）の例として、
    ピッチシフトやタイムストレッチを行う（任意）。
    """
    # 例: ランダムに ±2 半音までピッチシフト
    semitones = np.random.uniform(-2, 2)  
    audio_aug = librosa.effects.pitch_shift(audio, sr, n_steps=semitones)
    
    # 例: ランダムに ±10% のタイムストレッチ
    rate = np.random.uniform(0.9, 1.1)
    audio_aug = librosa.effects.time_stretch(audio_aug, rate)
    
    return audio_aug

def extract_features(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
                     fmin=FMIN, fmax=FMAX):
    """
    メルスペクトログラムを作成。
    fmin, fmax を指定して不要な周波数帯を除去し、対数変換。
    最終的に (周波数, 時間, 1) の形にする。
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,   # 最小周波数
        fmax=fmax    # 最大周波数
    )
    
    # 対数メルスペクトログラムへ
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # (周波数, 時間) -> (周波数, 時間, 1) へ reshape
    log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
    
    return log_mel_spec

def create_dataset(dataset_dir="dataset"):
    """
    dataset_dir 以下に「mix_voice」「not_mix_voice」フォルダを想定。
    それぞれのファイルを読み込み、特徴量とラベルを返す。
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
            # 1) 音声ファイルをロード
            audio = load_audio_file(file_path)
            
            # 2) データ拡張（オプション）
            if ENABLE_DATA_AUGMENT:
                audio = apply_data_augmentation(audio, SAMPLE_RATE)
            
            # 3) メルスペクトログラム抽出（fmin, fmax 指定）
            feat = extract_features(audio)
            
            X.append(feat)
            y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    return X, y

# =========================================================
# 3. CNNモデルの定義
# =========================================================
def create_cnn_model(input_shape):
    """
    CNN で (周波数, 時間, 1) の形状を入力とし、二値分類を行う。
    BatchNormalization を加えて学習を安定化。
    """
    model = keras.Sequential([
        # Conv2D -> BatchNorm -> ReLU (activation='relu')
        keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# =========================================================
# 4. 学習・評価の実行
# =========================================================
if __name__ == "__main__":
    # ---- (A) データセットの作成 ----
    X, y = create_dataset("dataset")
    print("X shape:", X.shape, "y shape:", y.shape)
    
    # 例: (サンプル数, n_mels, time_frames, 1)
    #     (サンプル数,)
    
    # ---- (B) Train/Test 分割 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )
    
    # ---- (C) 特徴量の標準化 ----
    # メルスペクトログラムの値はデータによってスケールが違う可能性があるので、
    # Flattenしてから StandardScaler を当て、また元の形に戻す方法がシンプル。
    shape_ = X_train.shape[1:]  # (n_mels, time_frames, 1)
    dim_ = shape_[0] * shape_[1] * shape_[2]  # 全特徴量数
    
    # Flatten
    X_train_2d = X_train.reshape(len(X_train), dim_)
    X_test_2d  = X_test.reshape(len(X_test), dim_)
    
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_test_2d  = scaler.transform(X_test_2d)
    
    # 元の形に戻す
    X_train = X_train_2d.reshape(len(X_train), *shape_)
    X_test  = X_test_2d.reshape(len(X_test), *shape_)
    
    # ---- (D) モデル構築 ----
    input_shape = X_train.shape[1:]  # (n_mels, time_frames, 1)
    model = create_cnn_model(input_shape)
    model.summary()
    
    # ---- (E) コールバック設定 (EarlyStopping) ----
    # 過学習を防ぐため、検証用データの損失が悪化し始めたら学習を止める
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # ---- (F) モデル学習 ----
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,   # 学習データのうち 20% をバリデーションに使用
        epochs=30,
        batch_size=16,
        callbacks=[early_stopping]
    )
    
    # ---- (G) テストデータで評価 ----
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"[Test] Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")
    
    # ---- (H) 推論のサンプル ----
    test_file_path = "sample_input.wav"  # 判定したい音声ファイル
    if os.path.exists(test_file_path):
        audio_test = load_audio_file(test_file_path)
        # fmin/fmax を使って特徴量抽出
        feat_test = extract_features(audio_test)
        
        # 形状 (n_mels, time_frames, 1) -> (1, n_mels, time_frames, 1)
        feat_test = np.expand_dims(feat_test, axis=0).astype(np.float32)
        
        # 同じ StandardScaler を適用
        feat_test_2d = feat_test.reshape(1, dim_)
        feat_test_2d = scaler.transform(feat_test_2d)
        feat_test = feat_test_2d.reshape(1, *shape_)
        
        pred = model.predict(feat_test)
        label = 1 if pred[0][0] >= 0.5 else 0
        score = float(pred[0][0])
        
        if label == 1:
            print(f"{test_file_path} はミックスボイスの可能性が高いです (score={score:.4f})")
        else:
            print(f"{test_file_path} はミックスボイスではない可能性が高いです (score={score:.4f})")
    else:
        print(f"推論用ファイルが見つかりません: {test_file_path}")
