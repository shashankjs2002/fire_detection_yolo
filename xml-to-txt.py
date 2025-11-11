import os
import xml.etree.ElementTree as ET

# Define your classes
classes = {'fire': 0, 'smoke': 1}

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_xml_to_txt(xml_folder, txt_folder):
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        in_file = open(os.path.join(xml_folder, xml_file))
        out_file = open(os.path.join(txt_folder, xml_file.replace('.xml', '.txt')), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes[cls]
            xmlbox = obj.find('bndbox')
            b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
            bb = convert_bbox((w, h), b)
            out_file.write(f"{cls_id} {' '.join([f'{a:.3f}' for a in bb])}\n")
        in_file.close()
        out_file.close()

# USAGE
convert_xml_to_txt("archive\\Annotations\\Annotations", "dataset\\labels")
