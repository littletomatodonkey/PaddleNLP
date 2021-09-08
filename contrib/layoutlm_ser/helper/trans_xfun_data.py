import json
# 替换成自己的输入输出路径即可
def transfer_xfun_data(json_path="/paddle/data/xfun/zh.train.json",
                     output_file="./xfun_normalize_train.json"):
    with open(json_path, "r") as fin:
        lines = fin.readlines()
    
    json_info = json.loads(lines[0])
    documents = json_info["documents"]
    label_info = {}
    with open(output_file, "w") as fout:
        for idx, document in enumerate(documents):
            img_info = document["img"]
            document = document["document"]
            image_path = "img/" + img_info["fname"]


            label_info["height"] = img_info["height"]
            label_info["width"] = img_info["width"]

            label_info["ocr_info"] = []

            for doc in document:
                label_info["ocr_info"].append({
                    "text": doc["text"],
                    "label": doc["label"],
                    "bbox": [1958, 144, 2184, 198],
                })
            
            fout.write(image_path + "\t" + json.dumps(label_info, ensure_ascii=False) + "\n")
    
    print("===ok====")


transfer_xfun_data()
