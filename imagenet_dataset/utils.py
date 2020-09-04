from pathlib import Path


def parse_label_id():
    label2id = {}
    name2label = {}
    with open("imagenet_clsidx_to_labels.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            name = line.split(",")[0].split(":")[1][2:]
            if name[-1] == "'" or name[-1] == '"':
                name = name[:-1]
            name2label[name] = int(line.split(":")[0])


    with open("./imagenet_labels_to_class.txt", "r") as f:
        lines = f.readlines()
        for l in lines:
            folder_name = l.split()[0]
            category_name = l.split()[2]
            category_name = category_name.replace("_", " ")

            label2id[folder_name] = name2label[category_name]

    return label2id


def main():
    parse_label_id()


if __name__ == "__main__":
    main()
