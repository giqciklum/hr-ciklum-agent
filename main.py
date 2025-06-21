import os, json, tempfile, shutil, datetime as dt
from googleapiclient.discovery import build
from google.cloud import storage
from google.oauth2 import service_account

# These are set by the deployment command
FOLDER_ID  = os.environ["DRIVE_FOLDER_ID"]
BUCKET     = os.environ["TARGET_BUCKET"] # This will be gs://hr-bot-docs
MANIFEST   = "manifest.json"             # A log file stored in the bucket

def _drive():
    """Initializes the Google Drive service."""
    return build("drive", "v3", cache_discovery=False)

def _list(folder_id):
    """Lists all files in a given Drive folder."""
    # This query finds all files directly inside the folder that are not in the trash 
    q = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None
    while True:
        resp = _drive().files().list(q=q,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)").execute()
        files += resp.get("files", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def sync_drive_to_local(request):
    """The main function that syncs Drive to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET.lstrip("gs://"))

    # Load the manifest of previously synced files from the bucket 
    blob = bucket.blob(MANIFEST)
    if blob.exists():
        manifest = json.loads(blob.download_as_text())
    else:
        manifest = {}

    print(f"Found {len(manifest)} files in existing manifest.")
    updated_files = {}

    # Create a temporary directory to download files into 
    with tempfile.TemporaryDirectory() as tmpdir:
        drive_files = _list(FOLDER_ID)
        print(f"Found {len(drive_files)} files in Google Drive folder.")

        for f in drive_files:
            # Skip sub-folders for now 
            if f["mimeType"] == "application/vnd.google-apps.folder":
                continue

            # Check if the file has been modified since the last sync
            modified_time = f["modifiedTime"][:19]
            if manifest.get(f["id"]) == modified_time:
                continue # File is unchanged, skip it

            print(f"Found updated file: {f['name']}")

            # Download the file from Drive 
            data = _drive().files().get_media(fileId=f["id"]).execute()
            dst = os.path.join(tmpdir, f["name"])
            with open(dst, "wb") as out:
                out.write(data)

            # Upload the file to our GCS bucket
            target_blob = bucket.blob(f["name"])
            target_blob.upload_from_filename(dst)

            updated_files[f["id"]] = modified_time

    # If any files were updated, save the new manifest back to the bucket 
    if updated_files:
        manifest.update(updated_files)
        blob.upload_from_string(json.dumps(manifest), content_type="application/json")
        msg = f"Synced {len(updated_files)} file(s) to {BUCKET}"
        print(msg)
        return msg

    print("No changes detected.")
    return "No changes"