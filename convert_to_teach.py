from admin_training import training_data

with open("teach_commands.txt", "w", encoding="utf-8") as file:
    for question, answer in training_data:
        file.write(f"teach: {question} => {answer}\n\n")

print("✅ teach_commands.txt created successfully!")