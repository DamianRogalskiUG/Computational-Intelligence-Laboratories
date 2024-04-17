import math
import datetime

def calculate_biorhythms(day_num):
    physical = math.sin(2 * math.pi * day_num / 23)
    emotional = math.sin(2 * math.pi * day_num / 28)
    intellectual = math.sin(2 * math.pi * day_num / 33)

    return [physical, emotional, intellectual]


def check_biorhythm_results(physical, emotional, intellectual):
    biorythms_result = {'physical': physical, 'emotional': emotional, 'intellectual': intellectual}
    for key, value in biorythms_result.items():
        if value > 0.5:
            print(f"Your result {key} is high ({value:.2f}). Congratulations!")
        elif value < -0.5:
            print(f"Your result {key} is low ({value:.2f}). Don't worry. Be happy!")
            next_results = calculate_biorhythms(current_day_number + 1)
            next_day_physical = next_results[0]
            next_day_emotional = next_results[1]
            next_day_intellectual = next_results[2]
            if next_day_physical > physical or next_day_emotional > emotional or next_day_intellectual > intellectual:
                print("Don't worry. Tomorrow you will get better!")


name = input("Input your name: ")
year = int(input("Input year of birth: "))
month = int(input("Input your month of birth: "))
day = int(input("Input your day of birth: "))

birthdate = datetime.date(year, month, day)
current_day_number = (datetime.date.today() - birthdate).days

results_tab = calculate_biorhythms(current_day_number)
physical = results_tab[0]
emotional = results_tab[1]
intellectual = results_tab[2]

print(f"Hello {name}!")
print(f"Today is {datetime.date.today().strftime('%d.%m.%Y')}, {current_day_number} day of your life.")
print("Your biorythms:")
print(f"- physical: {physical:.2f}")
print(f"- emotional: {emotional:.2f}")
print(f"- intellectual: {intellectual:.2f}")

check_biorhythm_results(physical, emotional, intellectual)

# bez AI trwalo to 25 min

# poprawiony kod prze ChatGPT

# import math
# import datetime
#
# def calculate_biorhythms(day_num):
#     physical = math.sin(2 * math.pi * day_num / 23)
#     emotional = math.sin(2 * math.pi * day_num / 28)
#     intellectual = math.sin(2 * math.pi * day_num / 33)
#
#     return [physical, emotional, intellectual]
#
#
# def check_biorhythm_results(physical, emotional, intellectual, current_day_number):
#     biorhythms_result = {'physical': physical, 'emotional': emotional, 'intellectual': intellectual}
#     for key, value in biorhythms_result.items():
#         if value > 0.5:
#             print(f"Your {key} biorhythm is high ({value:.2f}). Congratulations!")
#         elif value < -0.5:
#             print(f"Your {key} biorhythm is low ({value:.2f}). Don't worry. Be happy!")
#             next_results = calculate_biorhythms(current_day_number + 1)
#             next_day_physical, next_day_emotional, next_day_intellectual = next_results
#             if (next_day_physical > physical or
#                 next_day_emotional > emotional or
#                 next_day_intellectual > intellectual):
#                 print("Don't worry. Tomorrow you will get better!")
#
#
# def main():
#     name = input("Input your name: ")
#     year = int(input("Input your year of birth: "))
#     month = int(input("Input your month of birth: "))
#     day = int(input("Input your day of birth: "))
#
#     birthdate = datetime.date(year, month, day)
#     current_day_number = (datetime.date.today() - birthdate).days
#
#     physical, emotional, intellectual = calculate_biorhythms(current_day_number)
#
#     print(f"\nHello, {name}!")
#     print(f"Today is {datetime.date.today().strftime('%d.%m.%Y')}, {current_day_number} day of your life.")
#     print("Your biorhythms:")
#     print(f"- Physical: {physical:.2f}")
#     print(f"- Emotional: {emotional:.2f}")
#     print(f"- Intellectual: {intellectual:.2f}")
#
#     check_biorhythm_results(physical, emotional, intellectual, current_day_number)
#
# if __name__ == "__main__":
#     main()

# ChatGPT był w stanie napisać kod podobny do mojego w przeciągu 1 min, jako prompt podałem polecenie zadania