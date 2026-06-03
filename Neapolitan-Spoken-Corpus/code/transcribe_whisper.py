]# === LOAD INITIAL DATA ===
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# === HELPER: TRANSCRIBE FILE ===
def transcribe_file(filepath):
    with open(filepath, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=LANGUAGE
        )
    return response.text

# === PROCESS EACH FILE ===
total_time = 0
count = 0

for filename in sorted(os.listdir(FOLDER_PATH)):
    match = re.match(r"(\d+)\.m4a", filename)
    if not match:
        continue

    file_id = int(match.group(1))
    file_path = os.path.join(FOLDER_PATH, filename)

    # Find matching entry by ID
    entry = next((item for item in data if item["id"] == file_id), None)
    if entry is None:
        print(f"Skipping {file_id}.m4a – No matching entry found.")
        continue

    try:
        print(f"Transcribing {file_id}.m4a...")
        start_time = time.time()
        transcription = transcribe_file(file_path)
        duration = time.time() - start_time

        entry["file"] = file_path
        entry["transcription"] = transcription
        entry["transcription_time"] = duration

        total_time += duration
        count += 1

        print(f"→ Done in {duration:.2f} seconds")
    except Exception as e:
        print(f"Error transcribing {file_id}.m4a: {e}")

# === SAVE OUTPUT ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# === SUMMARY ===
if count > 0:
    avg_time = total_time / count
    print(f"\nFinished transcribing {count} files.")
    print(f"Total transcription time: {total_time:.2f} seconds")
    print(f"Average transcription time: {avg_time:.2f} seconds")
else:
    print("No files were transcribed.")