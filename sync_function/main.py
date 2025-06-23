import os
import io
import json
import tempfile
import google.auth
from googleapiclient.discovery import build
from google.cloud import storage

# --- Environment Variables ---
FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
BUCKET = os.environ["TARGET_BUCKET"]
MANIFEST = "manifest.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- MimeType Mapping for Google Workspace File Exports ---
# This dictionary tells the script how to convert Google files to Office formats
EXPORT_MIMETYPES = {
    'application/vnd.google-apps.document': {
        'type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'ext': '.docx'
    },
    'application/vnd.google-apps.spreadsheet': {
        'type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'ext': '.xlsx'
    },
    'application/vnd.google-apps.presentation': {
        'type': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'ext': '.pptx'
    }
}


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
    """The main Cloud Function to sync a Google Drive folder to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET.lstrip("gs://"))
    drive_service = _get_drive_service()
    
    manifest_blob = bucket.blob(MANIFEST)
    old_manifest = {}
    if manifest_blob.exists():
        try:
            old_manifest = json.loads(manifest_blob.download_as_text())
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {MANIFEST}. Starting fresh.")
    
    drive_files = _list_drive_files(drive_service, FOLDER_ID)
    print(f"Found {len(drive_files)} files in Google Drive folder.")
    
    current_drive_state = {f['id']: {'name': f['name'], 'modifiedTime': f['modifiedTime'], 'mimeType': f['mimeType']} for f in drive_files if f['mimeType'] != 'application/vnd.google-apps.folder'}
    
    old_ids = set(old_manifest.keys())
    current_ids = set(current_drive_state.keys())
    deleted_ids = old_ids - current_ids
    
    if deleted_ids:
        print(f"Found {len(deleted_ids)} file(s) to delete.")
        for file_id in deleted_ids:
            if file_id in old_manifest:
                file_to_delete = old_manifest[file_id]
                print(f"Deleting '{file_to_delete['name']}' from bucket.")
                blob_to_delete = bucket.blob(file_to_delete['name'])
                if blob_to_delete.exists():
                    blob_to_delete.delete()

    files_synced = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_id, file_info in current_drive_state.items():
            if file_id in old_manifest and old_manifest[file_id]['modifiedTime'] == file_info['modifiedTime']:
                continue

            print(f"Syncing updated file: {file_info['name']}")
            
            file_name = file_info['name']
            gdrive_mimetype = file_info['mimeType']
            
            # --- FINAL LOGIC TO HANDLE ALL FILE TYPES ---
            if gdrive_mimetype in EXPORT_MIMETYPES:
                # It's a Google Doc/Sheet/Slide, so we EXPORT it to its Office format
                export_format = EXPORT_MIMETYPES[gdrive_mimetype]
                request = drive_service.files().export_media(fileId=file_id, mimeType=export_format['type'])
                file_name += export_format['ext']
                print(f"Exporting '{file_info['name']}' as '{file_name}'")
                
                fh = io.BytesIO()
                downloader = io.BufferedWriter(fh)
                downloader.write(request.execute())
                data = fh.getvalue()
                downloader.close()

            elif gdrive_mimetype.startswith('application/vnd.google-apps'):
                # It's another Google file (Drawing, Form), export as PDF as a fallback
                request = drive_service.files().export_media(fileId=file_id, mimeType='application/pdf')
                file_name += '.pdf'
                print(f"Exporting '{file_info['name']}' as '{file_name}' (PDF fallback)")
                
                fh = io.BytesIO()
                downloader = io.BufferedWriter(fh)
                downloader.write(request.execute())
                data = fh.getvalue()
                downloader.close()

            else:
                # It's a regular binary file (PDF, DOCX, etc.), so we DOWNLOAD it directly
                request = drive_service.files().get_media(fileId=file_id)
                data = request.execute()

            dst = os.path.join(tmpdir, file_name)
            with open(dst, "wb") as out:
                out.write(data)
            
            target_blob = bucket.blob(file_name)
            target_blob.upload_from_filename(dst)
            
            current_drive_state[file_id]['name'] = file_name
            files_synced += 1

    if files_synced > 0 or len(deleted_ids) > 0:
        print(f"Updating manifest.json with {len(current_drive_state)} files.")
        manifest_blob.upload_from_string(json.dumps(current_drive_state, indent=2), content_type="application/json")
        msg = f"Sync complete. Updated: {files_synced}, Deleted: {len(deleted_ids)}."
        print(msg)
        return msg, 200

    print("No changes detected.")
    return "No changes", 200