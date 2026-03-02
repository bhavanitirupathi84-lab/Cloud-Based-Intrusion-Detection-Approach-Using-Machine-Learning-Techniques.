import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
    column_names = [
        'duration', 'protocol_type', 'service', 'flag',
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'attack_type', 'difficulty'
    ]
    print("Loading training data...")
    train_df = pd.read_csv(
        r'C:\Users\bhava\OneDrive\Desktop\intrusion_detection_project\data\raw\KDDTrain+.txt',
        names=column_names, header=None)
    print("Loading test data...")
    test_df = pd.read_csv(
        r'C:\Users\bhava\OneDrive\Desktop\intrusion_detection_project\data\raw\KDDTest+.txt',
        names=column_names, header=None)
    return train_df, test_df

def encode_categorical(df):
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col}")
    return df

def map_attack_types(df):
    attack_mapping = {
        'normal': 0,
        'guess_passwd': 1, 'ftp_write': 1,
        'warezclient': 2, 'warezmaster': 2, 'spy': 2, 'imap': 2,
        'portsweep': 3, 'ipsweep': 3, 'nmap': 3, 'satan': 3,
        'buffer_overflow': 4, 'loadmodule': 4, 'rootkit': 4, 'perl': 4,
        'multihop': 5, 'phf': 5,
        'back': 3, 'land': 3, 'neptune': 3, 'pod': 3,
        'smurf': 3, 'teardrop': 3
    }
    df['attack_category'] = df['attack_type'].map(attack_mapping)
    df['attack_category'] = df['attack_category'].fillna(3).astype(int)
    print("Attack types mapped!")
    return df

def engineer_features(df):
    df['duration_seconds'] = df['duration']
    df['is_long_connection'] = (df['duration'] > 60).astype(int)
    df['session_rate'] = df['count'] / (df['duration'] + 1)
    df['src_bytes_log'] = np.log1p(df['src_bytes'])
    df['dst_bytes_log'] = np.log1p(df['dst_bytes'])
    df['transfer_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
    df['connection_count'] = df['count']
    df['srv_connection_count'] = df['srv_count']
    df['connection_rate'] = df['count'] / (df['duration'] + 1)
    df['error_rate_total'] = df['serror_rate'] + df['rerror_rate']
    df['srv_error_rate_total'] = df['srv_serror_rate'] + df['srv_rerror_rate']
    df['is_logged_in'] = df['logged_in']
    df['failed_login_count'] = df['num_failed_logins']
    selected_features = [
        'duration_seconds', 'is_long_connection', 'session_rate',
        'src_bytes_log', 'dst_bytes_log', 'transfer_ratio',
        'connection_count', 'srv_connection_count', 'connection_rate',
        'error_rate_total', 'srv_error_rate_total',
        'is_logged_in', 'failed_login_count'
    ]
    print(f"Created {len(selected_features)} features!")
    return df, selected_features

if __name__ == "__main__":
    train_df, test_df = load_data()
    train_df = encode_categorical(train_df)
    test_df = encode_categorical(test_df)
    train_df = map_attack_types(train_df)
    test_df = map_attack_types(test_df)
    train_df, features = engineer_features(train_df)
    test_df, _ = engineer_features(test_df)
    print("ALL PREPROCESSING COMPLETE!")



def map_attack_types(df):
    """
    Map NSL-KDD attack types to our 5 categories
    
    0: normal
    1: brute_force
    2: data_exfiltration
    3: geo_anomaly (probe attacks)
    4: privilege_escalation (U2R attacks)
    5: insider_threat (R2L attacks)
    """
    print("\n" + "="*60)
    print("MAPPING ATTACK TYPES")
    print("="*60)
    
    print("\nOriginal attack types:")
    print(df['attack_type'].value_counts())
    
    # Define the mapping
    attack_mapping = {
        # Normal traffic
        'normal': 0,
        
        # Brute Force attacks
        'guess_passwd': 1,
        'ftp_write': 1,
        
        # Data Exfiltration
        'warezclient': 2,
        'warezmaster': 2,
        'spy': 2,
        'imap': 2,
        
        # Geo Anomaly (Probe attacks)
        'portsweep': 3,
        'ipsweep': 3,
        'nmap': 3,
        'satan': 3,
        
        # Privilege Escalation (U2R)
        'buffer_overflow': 4,
        'loadmodule': 4,
        'rootkit': 4,
        'perl': 4,
        
        # Insider Threat (R2L)
        'multihop': 5,
        'phf': 5,
        
        # DoS attacks → map to geo_anomaly
        'back': 3,
        'land': 3,
        'neptune': 3,
        'pod': 3,
        'smurf': 3,
        'teardrop': 3
    }
    
    # Apply the mapping
    df['attack_category'] = df['attack_type'].map(attack_mapping)
    
    # Handle any unmapped types (just in case)
    df['attack_category'].fillna(3, inplace=True)
    df['attack_category'] = df['attack_category'].astype(int)
    
    print("\nMapped to 6 categories:")
    print(df['attack_category'].value_counts().sort_index())
    
    # Show the distribution
    category_names = {
        0: 'normal',
        1: 'brute_force',
        2: 'data_exfiltration',
        3: 'geo_anomaly',
        4: 'privilege_escalation',
        5: 'insider_threat'
    }
    
    print("\nCategory breakdown:")
    for cat_id, cat_name in category_names.items():
        count = (df['attack_category'] == cat_id).sum()
        pct = count / len(df) * 100
        print(f"  {cat_id}: {cat_name:20s} - {count:6d} ({pct:5.2f}%)")
    
    return df

def map_attack_types(df):

    attack_mapping = {
        'normal': 0,

        'guess_passwd': 1, 'ftp_write': 1,

        'warezclient': 2, 'warezmaster': 2, 'spy': 2, 'imap': 2,

        'portsweep': 3, 'ipsweep': 3, 'nmap': 3, 'satan': 3,
        'back': 3, 'land': 3, 'neptune': 3, 'pod': 3,
        'smurf': 3, 'teardrop': 3,

        'buffer_overflow': 4, 'loadmodule': 4, 'rootkit': 4, 'perl': 4,

        'multihop': 5, 'phf': 5
    }

    df['attack_category'] = df['attack_type'].map(attack_mapping)

    # ✅ FIXED LINE
    df['attack_category'] = df['attack_category'].fillna(3).astype(int)

    print("Attack types mapped!")

    return df