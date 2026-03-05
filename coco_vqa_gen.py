"""从 COCO val2017 标注生成四类 VQA：existence / spatial / counting / attribute。"""

import argparse, json, os, random
from collections import defaultdict
from pycocotools.coco import COCO


def _rel_pos(a, b):
    ax, ay = a[0]+a[2]/2, a[1]+a[3]/2
    bx, by = b[0]+b[2]/2, b[1]+b[3]/2
    dx, dy = ax-bx, ay-by
    if abs(dx) > abs(dy): return "to the right of" if dx > 0 else "to the left of"
    return "below" if dy > 0 else "above"


def gen_existence(coco, img_anns, all_cats, n=2):
    out = []
    for iid, anns in img_anns.items():
        info = coco.imgs[iid]; w, h = info["width"], info["height"]
        present = {coco.cats[a["category_id"]]["name"] for a in anns}
        for a in anns[:n]:
            c = coco.cats[a["category_id"]]["name"]
            out.append(dict(image_id=iid, image_file=info["file_name"],
                question=f"Is there a {c} in this image? Answer yes or no.",
                answer="yes", task_type="existence", target_bbox=a["bbox"],
                target_category=c, gt_present=True, image_width=w, image_height=h))
        absent = [c for c in all_cats if c not in present]
        if absent:
            c = random.choice(absent)
            out.append(dict(image_id=iid, image_file=info["file_name"],
                question=f"Is there a {c} in this image? Answer yes or no.",
                answer="no", task_type="existence",
                target_bbox=[w*.25, h*.25, w*.5, h*.5],
                target_category=c, gt_present=False, image_width=w, image_height=h))
    return out


def gen_spatial(coco, img_anns, n=2):
    out = []
    for iid, anns in img_anns.items():
        if len(anns) < 2: continue
        info = coco.imgs[iid]; sa = sorted(anns, key=lambda a: a["area"], reverse=True)
        done = 0
        for i in range(len(sa)):
            if done >= n: break
            for j in range(i+1, len(sa)):
                if done >= n: break
                a, b = sa[i], sa[j]
                if a["area"] < 900 or b["area"] < 900: continue
                ca, cb = coco.cats[a["category_id"]]["name"], coco.cats[b["category_id"]]["name"]
                rel = _rel_pos(a["bbox"], b["bbox"])
                out.append(dict(image_id=iid, image_file=info["file_name"],
                    question=f"Is the {ca} {rel} the {cb}? Answer yes or no.",
                    answer="yes", task_type="spatial", target_bbox=a["bbox"],
                    target_category=ca, gt_present=True,
                    image_width=info["width"], image_height=info["height"]))
                done += 1
    return out


def gen_counting(coco, img_anns):
    out = []
    for iid, anns in img_anns.items():
        info = coco.imgs[iid]; cats = defaultdict(list)
        for a in anns: cats[coco.cats[a["category_id"]]["name"]].append(a)
        for c, ca in cats.items():
            cnt = len(ca)
            if cnt < 1 or cnt > 10: continue
            x0=min(a["bbox"][0] for a in ca); y0=min(a["bbox"][1] for a in ca)
            x1=max(a["bbox"][0]+a["bbox"][2] for a in ca); y1=max(a["bbox"][1]+a["bbox"][3] for a in ca)
            out.append(dict(image_id=iid, image_file=info["file_name"],
                question=f"How many {c}s are in this image? Answer with a number.",
                answer=str(cnt), task_type="counting",
                target_bbox=[x0, y0, x1-x0, y1-y0], target_category=c, gt_present=True,
                image_width=info["width"], image_height=info["height"]))
    return out


def gen_attribute(coco, img_anns):
    out = []
    for iid, anns in img_anns.items():
        if len(anns) < 2: continue
        info = coco.imgs[iid]; sa = sorted(anns, key=lambda a: a["area"], reverse=True)
        if sa[0]["area"] < 2 * sa[-1]["area"]: continue
        bc = coco.cats[sa[0]["category_id"]]["name"]
        sc = coco.cats[sa[-1]["category_id"]]["name"]
        if bc == sc: continue
        out.append(dict(image_id=iid, image_file=info["file_name"],
            question=f"Which is larger in this image, the {bc} or the {sc}?",
            answer=bc, task_type="attribute", target_bbox=sa[0]["bbox"],
            target_category=bc, gt_present=True,
            image_width=info["width"], image_height=info["height"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_images", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); random.seed(args.seed)

    coco = COCO(f"{args.coco_dir}/annotations/instances_val2017.json")
    all_cats = [c["name"] for c in coco.cats.values()]

    img_anns = {}
    for iid in sorted(coco.getImgIds()):
        a = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))
        if a: img_anns[iid] = a
        if len(img_anns) >= args.max_images: break

    s1, s2, s3, s4 = gen_existence(coco, img_anns, all_cats), gen_spatial(coco, img_anns), \
                      gen_counting(coco, img_anns), gen_attribute(coco, img_anns)
    all_s = s1 + s2 + s3 + s4; random.shuffle(all_s)
    print(f"生成 {len(all_s)} 条: exist={len(s1)} spatial={len(s2)} count={len(s3)} attr={len(s4)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/ced_vqa_dataset.jsonl", "w") as f:
        for s in all_s: f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
