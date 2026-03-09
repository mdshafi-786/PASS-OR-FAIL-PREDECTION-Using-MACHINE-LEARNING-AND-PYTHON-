import joblib
import numpy as np
import sys


def get_float_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input. Please enter a numeric value.")


def main():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run the notebook first.")
        sys.exit(1)

    print("=" * 40)
    print("  Pass or Fail Prediction Tool")
    print("=" * 40)

    study_hours = get_float_input("\nEnter Study Hours (0-12): ", 0, 12)
    prev_result = get_float_input("Enter Previous Result (0-100): ", 0, 100)
    attendance = get_float_input("Enter Attendance % (0-100): ", 0, 100)

    data = scaler.transform([[study_hours, prev_result, attendance]])
    prediction = model.predict(data)

    print("\n" + "=" * 40)
    if prediction[0] == 1:
        print("  ✅ Prediction: PASS")
    else:
        print("  ❌ Prediction: FAIL")
    print("=" * 40)


if __name__ == "__main__":
    main()
