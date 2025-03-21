import jsonlines
from collections import defaultdict


def clean_text(text):
    return text.replace(',', '').replace('.', '').strip().lower()

def contains_yes(text):
    return 'yes' in text.lower() or 'is socially acceptable' in text.lower() or 'are socially acceptable' in text.lower()

def contains_no(text):
    return 'no' in text.lower() or 'is not socially acceptable' in text.lower() or 'are not socially acceptable' in text.lower()

def contains_neither(text):
    return 'neither' in text.lower()


if __name__ == "__main__":
    MODEL_NAMES = ["llama3", "gemma", "exaone", "yi", "internlm", "aya", "seallm"]

    for NAME in MODEL_NAMES:
        print(f"Model: {NAME}")
        correct_country_count = 0
        correct_count = 0
        total_entries = 2633

        correct_per_country = defaultdict(int)
        total_per_country = defaultdict(int)

        input_file = f'{NAME}.jsonl'
        output_file = f'{NAME}_accuracy.jsonl'

        with jsonlines.open(input_file) as file:
            for line in file:
                country = line.get('Country', '')
                cleaned_text = clean_text(line.get(NAME, ''))

                if contains_yes(cleaned_text):
                    cleaned_text = "yes"
                if contains_no(cleaned_text):
                    cleaned_text = "no"
                if contains_neither(cleaned_text):
                    cleaned_text = "neutral"

                if cleaned_text == line.get('Gold Label', '').lower():
                    correct_count += 1

                if cleaned_text == line.get('Gold Label', '').lower():
                    correct_country_count += 1
                    correct_per_country[country] += 1  # Track per country
                total_per_country[country] += 1

        global_accuracy = correct_count / total_entries

        results = []
        results.append({
            'Country': "total",
            'Total': total_entries,
            'Correct': correct_count,
            'Accuracy': round(global_accuracy, 4)
        })

        for country, correct_count in correct_per_country.items():
            total_count = total_per_country[country]
            accuracy = correct_count / total_count if total_count > 0 else 0
            results.append({
                'Country': country,
                'Total': total_count,
                'Correct': correct_count,
                'Accuracy': round(accuracy, 4)
            })

        with jsonlines.open(output_file, mode='w') as writer:
            writer.write_all(results)

        print("Accuracy: ", round(global_accuracy, 4))
        print("\n==============================================\n")
