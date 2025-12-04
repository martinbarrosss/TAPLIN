import csv
import os
import glob

def get_unanswered_questions_from_file(filepath):
    """
    Analyzes a single CSV file and returns a set of questions that were not answered.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        set: A set of questions considered unanswered.
    """
    NON_ANSWER_PHRASES = [
        "no he encontrado", "no pude encontrar", "no se encontró",
        "no encuentro información", "lo siento, no puedo", "no tengo información"
    ]
    
    unanswered_questions = set()

    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile, delimiter=';')
            if 'question' not in reader.fieldnames or 'answer' not in reader.fieldnames:
                # Silently skip files with wrong format in 'all' mode, but notify in single mode.
                return None 

            for row in reader:
                question = row.get('question', '').strip()
                answer = row.get('answer', '').strip().lower()
                
                if not question:
                    continue

                is_unanswered = False
                if not answer:
                    is_unanswered = True
                else:
                    for phrase in NON_ANSWER_PHRASES:
                        if phrase in answer:
                            is_unanswered = True
                            break
                
                if is_unanswered:
                    unanswered_questions.add(question)
    except Exception:
        # Silently skip files that can't be read
        return None

    return unanswered_questions

def main():
    """
    Main function to drive the analysis of model output CSVs.
    """
    output_dir = 'testing/output'
    if not os.path.isdir(output_dir):
        print(f"Error: El directorio '{output_dir}' no fue encontrado.")
        return

    csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
    if not csv_files:
        print(f"No se encontraron archivos .csv en '{output_dir}'.")
        return

    print("Selecciona el archivo que quieres analizar:")
    for i, filepath in enumerate(csv_files):
        print(f"  {i + 1}: {os.path.basename(filepath)}")
    print(f"  {len(csv_files) + 1}: Todos")

    try:
        choice = int(input("\nIntroduce tu elección (número): ")) - 1
    except ValueError:
        print("Error: Entrada no válida. Debes introducir un número.")
        return

    if choice == len(csv_files): # "Todos" option
        print("\nAnalizando todos los archivos para encontrar preguntas sin respuesta comunes...")
        
        all_unanswered_sets = []
        for filepath in csv_files:
            unanswered_set = get_unanswered_questions_from_file(filepath)
            if unanswered_set is not None:
                all_unanswered_sets.append(unanswered_set)

        if not all_unanswered_sets:
            print("No se pudieron analizar los archivos o no contienen preguntas sin respuesta.")
            return

        # Find the intersection of all sets
        common_unanswered = set.intersection(*all_unanswered_sets)

        print("\n--- Preguntas Comunes Sin Respuesta en TODOS los Archivos ---")
        if common_unanswered:
            for question in sorted(list(common_unanswered)):
                print(f"  - {question}")
            print(f"\nTotal de preguntas comunes sin respuesta: {len(common_unanswered)}")
        else:
            print("No se encontraron preguntas sin respuesta que sean comunes a todos los archivos.")

    elif 0 <= choice < len(csv_files): # Single file option
        filepath = csv_files[choice]
        filename = os.path.basename(filepath)
        print(f"\nAnalizando '{filename}'...")

        unanswered_questions = get_unanswered_questions_from_file(filepath)

        if unanswered_questions is None:
            print("Error: El archivo no tiene el formato esperado (columnas 'question' y 'answer').")
            return
        
        print(f"\n--- Preguntas Sin Respuesta en '{filename}' ---")
        if unanswered_questions:
            for question in sorted(list(unanswered_questions)):
                print(f"  - {question}")
            print(f"\nTotal de preguntas sin respuesta: {len(unanswered_questions)}")
        else:
            print("Todas las preguntas tienen una respuesta satisfactoria en este archivo.")
    else:
        print("Error: Elección no válida.")

if __name__ == "__main__":
    main()