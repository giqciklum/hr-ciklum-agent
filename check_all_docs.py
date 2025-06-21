# check_all_docs.py
import os
import glob
import pypdf

print("--- Checking all PDF files in the 'docs' folder ---")

success_count = 0
fail_count = 0

# Loop through every PDF file in the docs folder
for file_path in glob.glob("docs/*.pdf"):
    try:
        pdf = pypdf.PdfReader(file_path)
        full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])

        # Check if the extracted text is empty
        if not full_text.strip():
            print(f"❌ FAILED: {os.path.basename(file_path)} - No text could be extracted.")
            fail_count += 1
        else:
            # Print the number of characters found as a measure of success
            print(f"✅ SUCCESS: {os.path.basename(file_path)} - Found {len(full_text)} characters.")
            success_count += 1

    except Exception as e:
        print(f"💥 ERROR: {os.path.basename(file_path)} - Could not be read. Error: {e}")
        fail_count += 1

print("\n--- Report Complete ---")
print(f"Readable documents: {success_count}")
print(f"Unreadable/Failed documents: {fail_count}")

if fail_count > 0:
    print("\nPlease replace the FAILED documents with clean, text-based PDF versions.")
else:
    print("\nAll documents appear to be readable!")