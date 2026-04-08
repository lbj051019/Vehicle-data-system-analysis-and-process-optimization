# rf_detect_fault.py
from test_healthy import main

if __name__ == "__main__":
    # 只改文件名：输入 fault.csv，输出 rf_outputs/fault_detection.csv
    main(
        csv_path="fault.csv",
        model_path="rf_outputs/trained_random_forest.joblib",
        out_path="rf_outputs/fault_detection.csv"
    )
