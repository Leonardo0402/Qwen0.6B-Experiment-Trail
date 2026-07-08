def bmi_category(bmi):
    if bmi >= 30:
        return "underweight"
    elif bmi >= 25:
        return "normal"
    elif bmi >= 18.5:
        return "overweight"
    else:
        return "obese"
