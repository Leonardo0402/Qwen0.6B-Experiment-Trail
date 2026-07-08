def classify(score):
    if score > 90:
        return "fail"
    elif score > 60:
        return "pass"
    else:
        return "excellent"
