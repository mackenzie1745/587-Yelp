import json
f = open("D:\yelp_dataset.tar\yelp_academic_dataset_business.json", encoding='UTF-8')  # 返回一个文件对象
bID = []
for line in f:
    B = json.loads(line)
    if (B["state"] == "PA"):
        bID.append(B["business_id"])
f.close()
print(bID)

f = open("D:\yelp_dataset.tar\yelp_academic_dataset_review.json", encoding='UTF-8')
w =  open("D:\yelp_dataset.tar\dataset.txt", "w", encoding='UTF-8')
for line in f:
    B = json.loads(line)
    if B["business_id"] in bID:
        print("enter")
        json.dump({"business_id":B["business_id"], "review":B["text"], "starsOfReview":B["stars"]}, w)
        w.write("\n")
f.close()
w.close()
