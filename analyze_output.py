
import csv
import os

def find_unanswered_questions(filepath):
    """
    Reads a CSV file and prints the questions that have no answer or where the model
    indicates it could not find an answer.

    Args:
        filepath (str): The path to the CSV file.
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo '{filepath}' no fue encontrado.")
        return

    # Phrases that indicate a non-answer
    NON_ANSWER_PHRASES = [
        "no he encontrado",
        "no pude encontrar",
        "no se encontró",
        "no encuentro información",
        "lo siento, no puedo",
        "no tengo información",
    ]

    print(f"Analizando el archivo: {os.path.basename(filepath)}...")
    
    unanswered_count = 0
    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile, delimiter=';')
            
            if 'question' not in reader.fieldnames or 'answer' not in reader.fieldnames:
                print("Error: El CSV debe contener las columnas 'question' y 'answer'.")
                return

            for row in reader:
                question = row.get('question', '').strip()
                answer = row.get('answer', '').strip().lower() # To lowercase for case-insensitive search
                
                is_unanswered = False
                if not answer:
                    is_unanswered = True
                else:
                    for phrase in NON_ANSWER_PHRASES:
                        if phrase in answer:
                            is_unanswered = True
                            break
                
                if question and is_unanswered:
                    print(f"  - {question}")
                    unanswered_count += 1
            
            if unanswered_count == 0:
                print("Todas las preguntas fueron contestadas satisfactoriamente.")
            else:
                print(f"\nTotal de preguntas no contestadas (o sin información): {unanswered_count}")

    except Exception as e:
        print(f"Ocurrió un error al procesar el archivo: {e}")

if __name__ == "__main__":
    file_to_analyze = 'testing/output/FrigoResultsDeep.csv'
    find_unanswered_questions(file_to_analyze)

