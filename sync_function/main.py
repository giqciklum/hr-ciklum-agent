import os
import json
import tempfile
import google.auth
from googleapiclient.discovery import build
from google.cloud import storage

FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
BUCKET = os.environ["TARGET_BUCKET"]
MANIFEST = "manifest.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def _get_drive_service():
    """Initializes the Google Drive service with the correct authentication scopes."""
    credentials, project = google.auth.default(scopes=SCOPES)
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return service

def _list_drive_files(drive_service, folder_id):
    """Lists all non-folder files in a given Drive folder, including Shared Drives."""
    q = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None
    while True:
        # These parameters are required to search within Shared Drives
        resp = drive_service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageToken=page_token,
            corpora="allDrives",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def sync_drive_to_local(request):
    """
    The main Cloud Function to sync a Google Drive folder to a GCS bucket,
    including additions, updates, and deletions.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET.lstrip("gs://"))
    drive_service = _get_drive_service()
    
    # --- Load Old Manifest ---
    manifest_blob = bucket.blob(MANIFEST)
    old_manifest = {}
    if manifest_blob.exists():
        try:
            old_manifest = json.loads(manifest_blob.download_as_text())
        except json.JSONDecodeError:
            print("Warning: Could not parse manifest.json. Starting fresh.")
    
    # --- Get Current State of Drive Folder ---
    drive_files = _list_drive_files(drive_service, FOLDER_ID)
    print(f"Found {len(drive_files)} files in Google Drive folder.")
    
    current_drive_state = {f['id']: {'name': f['name'], 'modifiedTime': f['modifiedTime']} for f in drive_files if f['mimeType'] != 'application/vnd.google-apps.folder'}
    
    # --- Handle Deletions ---
    # Find files that were in the old manifest but are NOT in Drive anymore.
    old_ids = set(old_manifest.keys())
    current_ids = set(current_drive_state.keys())
    deleted_ids = old_ids - current_ids
    
    if deleted_ids:
        print(f"Found {len(deleted_ids)} file(s) to delete.")
        for file_id in deleted_ids:
            # Check if file info exists in old manifest before accessing
            if file_id in old_manifest:
                file_to_delete = old_manifest[file_id]
                print(f"Deleting '{file_to_delete['name']}' from bucket.")
                blob_to_delete = bucket.blob(file_to_delete['name'])
                if blob_to_delete.exists():
                    blob_to_delete.delete()
    
    # --- Handle Additions/Updates ---
    files_synced = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_id, file_info in current_drive_state.items():
            # Check if file is new or modified
            if file_id not in old_manifest or old_manifest[file_id]['modifiedTime'] != file_info['modifiedTime']:
                print(f"Syncing updated file: {file_info['name']}")
                
                # Download file from Drive
                data = drive_service.files().get_media(fileId=file_id).execute()
                dst = os.path.join(tmpdir, file_info['name'])
                with open(dst, "wb") as out:
                    out.write(data)
                
                # Upload file to GCS
                target_blob = bucket.blob(file_info['name'])
                target_blob.upload_from_filename(dst)
                files_synced += 1

    # --- Save New Manifest ---
    # Only update the manifest if there were any changes (syncs or deletes)
    if files_synced > 0 or len(deleted_ids) > 0:
        print(f"Updating manifest.json with {len(current_drive_state)} files.")
        manifest_blob.upload_from_string(json.dumps(current_drive_state, indent=2), content_type="application/json")
        msg = f"Sync complete. Updated: {files_synced}, Deleted: {len(deleted_ids)}."
        print(msg)
        return msg, 200

    print("No changes detected.")
    return "No changes", 200