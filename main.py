import os
import json
import tempfile
import datetime as dt

# ## CHANGE HERE ## - Added google.auth for handling credentials and scopes
import google.auth
from googleapiclient.discovery import build
from google.cloud import storage

# These are set by the deployment command
FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
BUCKET = os.environ["TARGET_BUCKET"]
MANIFEST = "manifest.json"

# ## CHANGE HERE ## - Define the required scope to read Google Drive files
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def _get_drive_service():
    """
    ## CHANGE HERE ##
    Initializes the Google Drive service with the correct authentication scopes.
    """
    # Get the application default credentials and add the Drive scope
    credentials, project = google.auth.default(scopes=SCOPES)
    # Build the service object with the scoped credentials
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return service

def _list_drive_files(drive_service, folder_id):
    """
    ## CHANGE HERE ## - Now accepts the service object as an argument.
    Lists all files in a given Drive folder.
    """
    q = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None
    while True:
        resp = drive_service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageToken=page_token
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def sync_drive_to_local(request):
    """The main function that syncs Drive to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET.lstrip("gs://"))

    # ## CHANGE HERE ## - Initialize the Drive service once at the start
    drive_service = _get_drive_service()

    blob = bucket.blob(MANIFEST)
    if blob.exists():
        manifest = json.loads(blob.download_as_text())
    else:
        manifest = {}

    print(f"Found {len(manifest)} files in existing manifest.")
    updated_files = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # ## CHANGE HERE ## - Pass the initialized service to the list function
        drive_files = _list_drive_files(drive_service, FOLDER_ID)
        print(f"Found {len(drive_files)} files in Google Drive folder.")

        for f in drive_files:
            if f["mimeType"] == "application/vnd.google-apps.folder":
                continue

            modified_time = f["modifiedTime"][:19]
            if manifest.get(f["id"]) == modified_time:
                continue

            print(f"Found updated file: {f['name']}")

            # ## CHANGE HERE ## - Use the initialized service to download the file
            data = drive_service.files().get_media(fileId=f["id"]).execute()
            dst = os.path.join(tmpdir, f["name"])
            with open(dst, "wb") as out:
                out.write(data)

            target_blob = bucket.blob(f["name"])
            target_blob.upload_from_filename(dst)
            updated_files[f["id"]] = modified_time

    if updated_files:
        manifest.update(updated_files)
        blob.upload_from_string(json.dumps(manifest), content_type="application/json")
        msg = f"Synced {len(updated_files)} file(s) to {BUCKET}"
        print(msg)
        return msg, 200

    print("No changes detected.")
    return "No changes", 200