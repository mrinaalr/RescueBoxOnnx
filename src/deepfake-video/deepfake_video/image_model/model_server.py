import argparse
import csv
import logging
import warnings
from typing import TypedDict
from rb.lib.ml_service import MLService
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (DirectoryInput,
                                             EnumParameterDescriptor, EnumVal,
                                             FileResponse, InputSchema,
                                             InputType, ParameterSchema,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor)

warnings.filterwarnings("ignore")
# from BNN github
import model as model
import torch
from retinaface import RetinaFace

from sim_data import defaultDataset

logger = logging.getLogger(__name__)

APP_NAME = "deepfake_detection_video"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output file",
        input_type=InputType.DIRECTORY,
    )
    ckpt_schema = ParameterSchema(
        key="ckpt_path",
        label="Path to the model checkpoint",
        value=TextParameterDescriptor(default="weights/dffd_M_unfrozen.ckpt"),
    )
    bool_schema = ParameterSchema(
        key="disable_facecrop",
        label="Disable facecrop",
        value=EnumParameterDescriptor(
            default="False",
            enum_vals=[
                EnumVal(key="True", label="True"),
                EnumVal(key="False", label="False"),
            ],
        ),
    )
    return TaskSchema(
        inputs=[input_schema, output_schema], parameters=[ckpt_schema, bool_schema]
    )


class Inputs(TypedDict):
    input_dataset: DirectoryInput
    output_file: DirectoryInput


class Parameters(TypedDict):
    ckpt_path: str
    disable_facecrop: str


cfg = {
    "dataset_path": "datasets/demo",
    "resolution": 224,
    "ckpt": "weights/dffd_M_unfrozen.ckpt",
}


def predict(net, sample, device, dataset, disable_facecrop=False):
    image = None
    if not disable_facecrop:
        faces = RetinaFace.extract_faces(
            sample["image_path"][:-1], expand_face_area=100
        )
        if len(faces) > 0:
            image = dataset.apply_transforms(faces[0])
        else:
            image = sample["image"]
    else:
        image = sample["image"]
    image = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logger.debug(f"I AM HERE HELLO")        
        import pdb; pdb.set_trace()
        output = net(image)
        logit = output["logits"][0][0]
        temp = torch.sigmoid(logit)
        pred = 1 if temp > 0.5 else 0
        conf = temp if temp > 0.5 else 1 - temp
        return pred, conf


@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    cfg["dataset_path"] = inputs["input_dataset"].path
    out = inputs["output_file"].path
    out = f"{out}\\predictions_" + str(int(torch.rand(1) * 1000)) + ".csv"
    data = defaultDataset(
        dataset_path=cfg["dataset_path"], resolution=cfg["resolution"]
    )
    print(parameters)
    disable_facecrop = parameters["disable_facecrop"] == "True"
    net = model.BNext4DFR.load_from_checkpoint(cfg["ckpt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    res_list = []
    for i in range(len(data)):
        sample = data[i]
        if sample is None:
            res_list.append(
                {"image_path": sample["image_path"][:-1], "prediction": "error"}
            )
            continue
        pred, conf = predict(net, sample, device, data, disable_facecrop)
        pred = (
            "likely real"
            if pred == 1 and conf > 0.8
            else "likely fake" if conf > 0.8 else "uncertain"
        )
        res_list.append(
            {
                "image_path": sample["image_path"][:-1],
                "prediction": pred,
                "confidence": conf.item(),
            }
        )

    with open(out, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_path", "prediction", "confidence"]
        )
        writer.writeheader()  # Write header row
        writer.writerows(res_list)  # Write data rows

    return ResponseBody(FileResponse(path=out, file_type="csv"))

# Create a server instance
server = MLService(APP_NAME)

info_file_path = Path(__file__).resolve().parent / "img-app-info.md"
with open(info_file_path, "r", encoding="utf-8") as f:
    app_info = f.read()

server.add_app_metadata(
    name="Video DeepFake Detector",
   author="UMass Rescue - Mrinaal",
    version="2.1.0",
    info=app_info,
    plugin_name=APP_NAME,
    gpu=True,
)


server.add_ml_service(
    rule="/predict",
    ml_function=give_prediction,
    inputs_cli_parser=typer.Argument(
        parser=cli_parser,
        help="Provide the input dataset directory and output file path.",
    ),
    parameters_cli_parser=typer.Argument(
        parser=param_parser,
        help="Comma-separated list of models to use (e.g., 'BNext_M_ModelONNX').",
    ),
    short_title="DeepFake Detection",
    order=0,
    task_schema_func=create_transform_case_task_schema,
)

app = server.app









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument(
        "--port", type=int, help="Port number to run the server", default=5000
    )
    args = parser.parse_args()
    print(
        "CUDA is available." if torch.cuda.is_available() else "CUDA is not available."
    )
    server.run(port=args.port)
