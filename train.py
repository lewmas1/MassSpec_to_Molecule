import subprocess
import os
def create_input(template_path, output_path, src_train_path, tgt_train_path, src_val_path, tgt_val_path):
    with open(template_path, "r") as f:
        template = f.read()

    src_vocab_path = "data/vocab/vocab.src"
    tgt_vocab_path = "data/vocab/vocab.tgt"
    save_model_path = "model"

    input_file = template.format(
        'data',
        src_vocab_path,
        tgt_vocab_path,
        src_train_path,
        tgt_train_path,
        src_val_path,
        tgt_val_path,
        'data',
        save_model_path
    )

    input_file_path = "input.yaml"
    with open(input_file_path, "w") as f:
        f.write(input_file)

    return input_file_path


def gen_vocab(log_path, input_file_path):
    with open("vocab.log", "w") as out:
        subprocess.call(
            ["onmt_build_vocab", "-config", input_file_path, "-n_sample", "-1"],
            stdout=out,
            stderr=out,
        )


def main(template_path, data_folder):
    log_path = "logs"
    os.makedirs(log_path, exist_ok=True)

    data_path = "data"
    src_train_path = os.path.join(data_path, "src-train.txt")
    tgt_train_path = os.path.join(data_path, "tgt-train.txt")
    src_val_path = os.path.join(data_path, "src-val.txt")
    tgt_val_path = os.path.join(data_path, "tgt-val.txt")

    input_file_path = create_input(
        template_path,
        data_folder,
        src_train_path,
        tgt_train_path,
        src_val_path,
        tgt_val_path,
    )

    gen_vocab(log_path, input_file_path)

    input_file_path = "input.yaml"

    train_logs_path = os.path.join(log_path, "train")
    out_train = os.path.join(train_logs_path, "out.txt")
    err_train = os.path.join(train_logs_path, "err.txt")

    os.makedirs(train_logs_path, exist_ok=True)

    with open(out_train, "w") as out_file, open(err_train, "w") as err_file:
        subprocess.call(
            ["onmt_train", "-config", input_file_path], stdout=out_file, stderr=err_file
        )


if __name__ == "__main__":
    template_path = "templates/transformer_template_cpu.yaml"
    data_folder = "data"
    main(template_path, data_folder)
