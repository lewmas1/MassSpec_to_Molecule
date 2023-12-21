import os
import subprocess

def test_model(test_data_path, model_path):
    test_logs_path = os.path.join(os.getcwd(), "test_logs")
    out_test = os.path.join(test_logs_path, "out.txt")
    err_test = os.path.join(test_logs_path, "err.txt")

    os.makedirs(test_logs_path, exist_ok=True)

    with open(out_test, "w") as out_file, open(err_test, "w") as err_file:
        subprocess.call(
            ["conda", "run", "-n", "base", "onmt_translate", "-model", model_path, "-src", test_data_path, "-output", "pred.txt"],
            stdout=out_file,
            stderr=err_file
        )

if __name__ == "__main__":
    test_data_path = "data/src-test.txt"
    model_path = "model/model_step_5.pt"
    test_model(test_data_path, model_path)
