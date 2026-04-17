from google.cloud import aiplatform

aiplatform.init(
    project="nlp-research-sp26",
    location="us-central1",
    staging_bucket="gs://project3102-model-bucket"
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="mcberth-pretrain-v1-test",
    container_uri="us-docker.pkg.dev/nlp-research-sp26/mcberth-training/mcberth-training:v1-test",
)

job.run(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    replica_count=1,
    base_output_dir="gs://project3102-model-bucket/Training-Tests/McBERTh-Pretrain-v1-test",
)

print("Job submitted successfully!")