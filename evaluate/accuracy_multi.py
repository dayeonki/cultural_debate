import jsonlines
from collections import defaultdict
from utils import country_iso_mapping


def clean_text(text):
    return text.replace(',', '').replace('.', '').strip().lower()

def contains_yes(text):
    return 'yes' in text.lower() or 'is socially acceptable' in text.lower() or 'are socially acceptable' in text.lower()

def contains_no(text):
    return 'no' in text.lower() or 'is not socially acceptable' in text.lower() or 'are not socially acceptable' in text.lower()

def contains_neither(text):
    return 'neither' in text.lower()


if __name__ == "__main__":
    FIRST_MODEL = "aya"
    SECOND_MODEL = "seallm"

    correct_iso_count_1 = 0
    correct_iso_count_2 = 0
    correct_count_1 = 0
    correct_count_2 = 0
    total_entries = 2633

    correct_per_iso_1 = defaultdict(int)
    total_per_iso_1 = defaultdict(int)
    correct_per_iso_2 = defaultdict(int)
    total_per_iso_2 = defaultdict(int)

    input_file = f'{FIRST_MODEL}_{SECOND_MODEL}.jsonl'
    output_file = f'{FIRST_MODEL}_{SECOND_MODEL}_accuracy.jsonl'

    # Compute accuracy for first model's final decision
    with jsonlines.open(input_file) as file:
        for line in file:
            country = line.get('Country', 'unknown')
            iso_code = country_iso_mapping.get(country)

            cleaned_text1 = clean_text(line.get(f'{FIRST_MODEL}_final', ''))
            if contains_yes(cleaned_text1):
                cleaned_text1 = "yes"
            if contains_no(cleaned_text1):
                cleaned_text1 = "no"
            if contains_neither(cleaned_text1):
                cleaned_text1 = "neutral"

            if cleaned_text1 == line.get('Gold Label', '').lower():
                correct_count_1 += 1

            if cleaned_text1 == line.get('Gold Label', '').lower():
                correct_iso_count_1 += 1
                correct_per_iso_1[iso_code] += 1  # Track per ISO code
            total_per_iso_1[iso_code] += 1

    # Compute accuracy for second model's final decision
    with jsonlines.open(input_file) as file:
        for line in file:
            country = line.get('Country', 'unknown')
            iso_code = country_iso_mapping.get(country)

            cleaned_text2 = clean_text(line.get(f'{SECOND_MODEL}_final', ''))
            if contains_yes(cleaned_text2):
                cleaned_text2 = "yes"
            if contains_no(cleaned_text2):
                cleaned_text2 = "no"
            if contains_neither(cleaned_text2):
                cleaned_text2 = "neutral"

            if cleaned_text2 == line.get('Gold Label', '').lower():
                correct_count_2 += 1

            if cleaned_text2 == line.get('Gold Label', '').lower():
                correct_iso_count_2 += 1
                correct_per_iso_2[iso_code] += 1  # Track per ISO code
            total_per_iso_2[iso_code] += 1

    global_accuracy_1 = correct_count_1 / total_entries
    global_accuracy_2 = correct_count_2 / total_entries

    results = []
    results.append({
        'Country': "total",
        'Total': total_entries,
        'Accuracy1': round(global_accuracy_1, 4),
        'Accuracy2': round(global_accuracy_2, 4)
    })

    for iso_code, correct_count in correct_per_iso_1.items():
        total_count = total_per_iso_1[iso_code]
        accuracy = correct_count_1 / total_count if total_count > 0 else 0
        results.append({
            'Model': FIRST_MODEL,
            'ISO': iso_code,
            'Total': total_count,
            'Correct': correct_count_1,
            'Accuracy': round(accuracy, 4)
        })

    for iso_code, correct_count in correct_per_iso_2.items():
        total_count = total_per_iso_2[iso_code]
        accuracy = correct_count_2 / total_count if total_count > 0 else 0
        results.append({
            'Model': SECOND_MODEL,
            'ISO': iso_code,
            'Total': total_count,
            'Correct': correct_count_2,
            'Accuracy': round(accuracy, 4)
        })

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(results)

    print(f"{FIRST_MODEL} Accuracy: ", round(global_accuracy_1, 4))
    print(f"{SECOND_MODEL} Accuracy: ", round(global_accuracy_2, 4))
