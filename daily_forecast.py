import os
import numpy as np
import pandas as pd
import requests
import joblib
import firebase_admin
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from firebase_admin import credentials, db
from datetime import datetime

# Import TensorFlow
import tensorflow as tf
from tensorflow import keras

# --- 1. CẤU HÌNH ---
SHEET_NAME = 'ESP32' 
WORKSHEET_NAME = 'Sheet1' 

DATABASE_URL = 'https://test-weather-station-default-rtdb.firebaseio.com/' 
LAT = 10.8231
LON = 106.6297
HISTORY_DAYS = 30 
FEATURE_COLS = ['Nhiệt độ', 'Độ ẩm', 'Áp suất', 'Tốc độ gió', 'Hướng gió', 'Lượng mưa']

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'weather_forecast_model.h5')
scaler_in_path = os.path.join(base_dir, 'scaler_features.joblib')
scaler_out_path = os.path.join(base_dir, 'scaler_targets_continuous.joblib')
key_path = os.path.join(base_dir, 'serviceAccountKey.json')

# --- HÀM CHUYỂN ĐỔI HƯỚNG GIÓ ---
def convert_wind_direction(direction_str):
    try:
        d = str(direction_str).lower().strip()
        mapping = {
            'bắc': 0, 'b': 0, 'north': 0, 'n': 0,
            'đông bắc': 45, 'đb': 45, 'ne': 45,
            'đông': 90, 'đ': 90, 'east': 90, 'e': 90,
            'đông nam': 135, 'đn': 135, 'se': 135,
            'nam': 180, 'n': 180, 'south': 180, 's': 180,
            'tây nam': 225, 'tn': 225, 'sw': 225,
            'tây': 270, 't': 270, 'west': 270, 'w': 270,
            'tây bắc': 315, 'tb': 315, 'nw': 315
        }
        return mapping.get(d, 0)
    except:
        return 0

# --- 2. HÀM LẤY DỮ LIỆU NỀN (OPEN-METEO) ---
def get_open_meteo_backup():
    print("🌐 Đang tải dữ liệu nền từ Open-Meteo (Backup)...")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,rain,surface_pressure,wind_speed_10m,wind_direction_10m&past_days=40&forecast_days=1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        hourly = data['hourly']
        df = pd.DataFrame({
            'Ngày': pd.to_datetime(hourly['time']),
            'Nhiệt độ': hourly['temperature_2m'],
            'Độ ẩm': hourly['relative_humidity_2m'],
            'Áp suất': hourly['surface_pressure'],
            'Tốc độ gió': hourly['wind_speed_10m'],
            'Hướng gió': hourly['wind_direction_10m'],
            'Lượng mưa': hourly['rain']
        })
        df.set_index('Ngày', inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Không gọi được Open-Meteo: {e}")
        return None

# --- 3. HÀM LẤY DỮ LIỆU TỪ GOOGLE SHEET ---
def get_google_sheet_data():
    print("☁️ Đang tải dữ liệu từ Google Sheet (ESP32)...")
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)
        data = sheet.get_all_records()
        
        if not data: return None

        df = pd.DataFrame(data)
        
        try:
            df['DateTimeStr'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
            df['Ngày'] = pd.to_datetime(df['DateTimeStr'], errors='coerce')
        except Exception as e:
            print(f"⚠️ Lỗi xử lý ngày tháng Sheet: {e}")
            return None
            
        rename_map = {
            'Temperature': 'Nhiệt độ',
            'Humidity': 'Độ ẩm',
            'Pressure': 'Áp suất',
            'Wind Speed': 'Tốc độ gió',
            'Wind Direction': 'Hướng gió',
            'Rainfall': 'Lượng mưa'
        }
        df.rename(columns=rename_map, inplace=True)
        
        if 'Hướng gió' in df.columns:
            df['Hướng gió'] = df['Hướng gió'].apply(convert_wind_direction)
            
        df = df.dropna(subset=['Ngày'])
        df.set_index('Ngày', inplace=True)
        
        cols_numeric = ['Nhiệt độ', 'Độ ẩm', 'Áp suất', 'Tốc độ gió', 'Hướng gió', 'Lượng mưa']
        for col in cols_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[df['Nhiệt độ'] > 0] 
        df = df[cols_numeric]
        
        print(f"✅ Đã tải {len(df)} dòng dữ liệu từ Sheet.")
        return df
        
    except Exception as e:
        print(f"⚠️ Lỗi đọc Google Sheet: {e}")
        return None

# --- 4. HÀM TRỘN DỮ LIỆU ---
def get_hybrid_data():
    df_meteo = get_open_meteo_backup()
    if df_meteo is None: return None
    
    df_esp32 = get_google_sheet_data()
    
    if df_esp32 is not None and not df_esp32.empty:
        print("⚡ Đang ghép nối: ESP32 + Open-Meteo...")
        df_esp32_hourly = df_esp32.resample('H').mean()
        df_merged = df_esp32_hourly.combine_first(df_meteo)
    else:
        print("⚠️ Dùng 100% dữ liệu Open-Meteo.")
        df_merged = df_meteo
        
    df_daily = df_merged.resample('D').mean().dropna()
    
    if len(df_daily) < HISTORY_DAYS:
        print(f"❌ Không đủ dữ liệu (Có {len(df_daily)} ngày).")
        return None
        
    df_final = df_daily.iloc[-HISTORY_DAYS:][FEATURE_COLS]
    print(f"✅ Dữ liệu đầu vào: {df_final.index[0].date()} -> {df_final.index[-1].date()}")
    return df_final.values

# --- 5. CHẠY DỰ BÁO ---
def run_forecast():
    print("\n--- BẮT ĐẦU QUÁ TRÌNH DỰ BÁO ---")
    print("📥 Đang load Model & Scaler...")
    try:
        model = keras.models.load_model(model_path, compile=False)
        scaler_features = joblib.load(scaler_in_path)
        scaler_targets = joblib.load(scaler_out_path)
    except Exception as e:
        print(f"❌ Lỗi load file: {e}")
        return

    raw_data = get_hybrid_data()
    if raw_data is None: return
    
    input_scaled = scaler_features.transform(np.array(raw_data))
    current_window = input_scaled.reshape(1, HISTORY_DAYS, 6)
    
    firebase_results = {}
    print("\n🔮 KẾT QUẢ DỰ BÁO 7 NGÀY TỚI (CHỈ 1 KIỂU THỜI TIẾT):")
    print("="*85)
    
    for i in range(7):
        try:
            pred_raw = model.predict(current_window, verbose=0)
            pred_flat = np.array(pred_raw).flatten()
            last_6_values = pred_flat[-6:] 
            
            continuous_part = last_6_values[:3]
            boolean_part = last_6_values[3:] # [Score_Nắng, Score_Mưa, Score_Giông]
            
            real_continuous = scaler_targets.inverse_transform([continuous_part])[0]
            val_nhiet = float(real_continuous[0])
            val_am = float(real_continuous[1])
            val_mua = float(real_continuous[2])
            if val_mua < 0: val_mua = 0

            # --- LOGIC MỚI: CHỌN 1 TRONG 3 (MAX SCORE) ---
            # Tìm xem chỉ số nào (0, 1 hay 2) có điểm cao nhất
            max_idx = np.argmax(boolean_part)
            
            # Reset tất cả về False
            is_nang = False
            is_mua = False
            is_giong = False
            
            icon_str = ""
            
            # Gán True cho người chiến thắng
            # Giả định thứ tự là [Nắng, Mưa, Giông]
            if max_idx == 0:
                is_nang = True
                icon_str = "☀️ Trời Nắng"
            elif max_idx == 1:
                is_mua = True
                icon_str = "🌧️ Trời Mưa"
            elif max_idx == 2:
                is_giong = True
                icon_str = "⛈️ Có Giông"

            day_key = f"Day_{i+1}"
            firebase_results[day_key] = {
                "nhietDo": round(val_nhiet, 1),
                "doAm": round(val_am, 1),
                "luongMua": round(val_mua, 2),
                "troiNang": is_nang,
                "troiMua": is_mua,
                "troiGiong": is_giong
            }
            
            # Hiển thị Terminal gọn gàng
            print(f"📅 {day_key}: "
                  f"🌡️ {val_nhiet:.1f}°C  |  "
                  f"💧 {val_am:.1f}%  |  "
                  f"🌧️ {val_mua:.2f}mm  |  "
                  f"{icon_str}")

            new_row = current_window[0, -1].copy()
            new_row[0] = continuous_part[0]
            new_row[1] = continuous_part[1]
            new_row[5] = continuous_part[2]
            current_window = np.append(current_window[:, 1:, :], [[new_row]], axis=1)
            
        except Exception as e:
            print(f"❌ Lỗi ngày {i+1}: {e}")
            return

    print("="*85)
    print("📤 Đang gửi dữ liệu lên Firebase...")
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
    
    ref = db.reference('weather_forecast')
    ref.set(firebase_results)
    print("✅ HOÀN TẤT!")

if __name__ == "__main__":
    run_forecast()