from sagemaker import s3
from sagemaker.pytorch import PyTorchModel
from sagemaker.deserializers import JSONDeserializer
from sagemaker import get_execution_role, Session
from datetime import datetime

# Initialize the SageMaker session
sess = Session()

# Get the SageMaker execution role
role = get_execution_role()

# Define the S3 location of the model data
bucket = "s3://pricing-algorithm-deployment"
model_data = f"{bucket}/model.tar.gz"

model_name = "yolov8l.pt"

model = PyTorchModel(
    entry_point="inference.py",
    model_data=model_data,
    framework_version="1.12",
    py_version="py38",
    role=role,
    sagemaker_session=sess,
)

print("Model created")

INSTANCE_TYPE = "m4.large"
ENDPOINT_NAME = "pricing_algorithm-" + str(
    datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    deserializer=JSONDeserializer(),
    endpoint_name=ENDPOINT_NAME,
)

print("Model deployed")
