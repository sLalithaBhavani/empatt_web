from google.cloud import storage
from google.oauth2 import service_account



def download_blob(bucket_name, source_file_name, dest_filename):
  bucket_name = 'emp_attendance_monitoring_processed'
  client = storage.Client.from_service_account_json("cloudkarya-internship-415b6b4ef0ff.json")
  bucket = client.get_bucket(bucket_name)
  storage_client = storage.Client.from_service_account_json("cloudkarya-internship-415b6b4ef0ff.json")
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(source_file_name)
  return blob.name
  #f = open(dest_filename,'wb')
  #blob.download_to_file(f)

