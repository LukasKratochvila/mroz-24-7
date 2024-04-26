import argparse
import os

import xml.etree.ElementTree as ET
import cv2
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

from utils import ind2rgb


def create_overlay(sub_dir, data_dir='', img_name='input.jpg', mask_name='mask.png', save=True,
                   alpha=0.6, color_map='floorplan_map'):
    path = os.path.join(data_dir, sub_dir)

    image = Image.open(os.path.join(path, img_name)).convert('RGB')
    mask = Image.open(os.path.join(path, mask_name))

    mask = Image.fromarray(ind2rgb(np.array(mask), color_map=color_map)).resize(image.size)

    overlay = Image.blend(image, mask, alpha)
    if save:
        overlay.save(os.path.join(path, '_'.join([img_name, mask_name, 'overlay.png'])))
    else:
        overlay.show()


def is_number(s: str):
    try:
        float(s.replace('.', '').replace('-', '').replace(',', '.'))
        return True
    except ValueError:
        return False


def transform_tofloat(x, y, matrix):
    x_o = float(matrix[0]) * float(x) + float(matrix[1]) * float(y) + float(matrix[4])
    y_o = float(matrix[2]) * float(x) + float(matrix[3]) * float(y) + float(matrix[5])
    return x_o, y_o


def draw(d: str, matrix):
    coord = []
    params_list = d.split(' ')
    keys = [params_list[0]]
    values = []
    for i in range(1, len(params_list)):
        if is_number(params_list[i]) and is_number(params_list[i - 1]):
            keys.append(keys[-1])
            values.append(params_list[i])
        elif is_number(params_list[i]):
            values.append(params_list[i])
        else:
            keys.append(params_list[i])

    x, y = values[0].split(',')
    coord.append([float(x), float(y)])

    for k, i in zip(keys[1:], range(1,len(keys))):
        x, y = coord[-1]

        if k == 'v':
            coord.append([x, y + float(values[i])])
        if k == 'h':
            coord.append([x + float(values[i]), y])
        if k == 'l' or k == 'm':
            dx, dy = values[i].split(',')
            coord.append([x + float(dx), y + float(dy)])
        if k == 'V':
            coord.append([x, float(values[i])])
        if k == 'H':
            coord.append([float(values[i]), y])
        if k == 'L' or k == 'M':
            x, y = values[i].split(',')
            coord.append([float(x), float(y)])
        if k == 'c' or k == 'C':
            pass
    if len(keys) > 2 and 'c' not in keys and 'C' not in keys:
        for i in range(len(keys)-1, 0, -1):
            coord.append(coord[i])
    coord.append(coord[0])

    if matrix is not None:
        points = list()
        for point in coord:
            points.append(transform_tofloat(point[0], point[1], matrix))
        coord = points

    return coord


def parseSVG(floor, elements, matrix):  # '#003fff' door  '#ff7f00' wall '#00bfff' window
    for child in floor[0]:
        if 'style' in child.attrib:
            if (child.attrib['style'].split(';')[1].split(':')[1] == '#ff7f00' or
                    child.attrib['style'].split(';')[1].split(':')[1] == '#dd3700'):  # Wall
                elements['Walls'].append(draw(child.attrib['d'], matrix))
            if child.attrib['style'].split(';')[1].split(':')[1] == '#003fff':  # Door
                elements['Doors'].append(draw(child.attrib['d'], matrix))
            if child.attrib['style'].split(';')[1].split(':')[1] == '#00bfff':  # Window
                elements['Windows'].append(draw(child.attrib['d'], matrix))
            if 'stroke:#7fff00' in child.attrib['style'] and 'd' in child.attrib.keys():  # Stairs
                elements['Stairs_all'].append(draw(child.attrib['d'], matrix))


def parse_label(path, img_name, svg_name, pbar):
    xml = ET.parse(os.path.join(path, svg_name))
    root = xml.getroot()

    classes = ['Walls', 'Glass_walls', 'Railings', 'Doors', 'Sliding_doors', 'Windows', 'Stairs_all']
    elements = {}
    for c in classes:
        elements[c] = []

    if 'transform' in root.find('{http://www.w3.org/2000/svg}g').attrib.keys():
        matrix = root.find('{http://www.w3.org/2000/svg}g').attrib['transform'].strip('matrix()').split(',')
    else:
        matrix = None

    pbar.set_description('Parsing svg file ...')
    for group in root.find('{http://www.w3.org/2000/svg}g'):
        parseSVG(group, elements, matrix)
    pbar.set_description('Parsing Done.')

    pbar.set_description('Drawing elements ...')
    h, w, _ = cv2.imread(os.path.join(path, img_name)).shape
    masks = {}
    for c in classes:
        bg = Image.new('L', size=(w, h))
        image_draw = ImageDraw.Draw(bg)
        for e in elements[c]:
            if len(e) == 1:  # Circle
                image_draw.ellipse(e, fill='white', outline='white', width=0)
            else:  # Polygon
                image_draw.line([(xy[0], xy[1]) for xy in e], fill='white', width=1)
        masks[c] = np.array(bg, np.uint8)
    pbar.set_description('Drawing Done.')

    pbar.set_description('Creating mask ...')
    mask = np.zeros((h, w, 1), np.uint8)
    for c in range(len(classes)):
        c_ind = masks[classes[c]]
        mask[c_ind == 255] = c + 1  # border
        if classes[c] in ['Windows']:
            c_wall = masks['Walls_filled']
            c_ind += c_wall
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(c_ind, flood_mask, (0, 0), 255)
            c_ind += c_wall
        else:
            flood_mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(c_ind, flood_mask, (0, 0), 255)
        mask[cv2.bitwise_not(c_ind) == 255] = c + 1  # fill
        masks[classes[c] + '_filled'] = cv2.bitwise_not(c_ind)
    cv2.imwrite(path + '/mask.png', mask)
    pbar.set_description('Mask Done.')

    pbar.set_description('Creating overlay ...')
    create_overlay('', path, img_name, 'mask.png', alpha=0.8)
    pbar.set_description('Overlay Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create mask for Projart Dataset from pdf file (*_farba) and'
                                                 'create overlay on the source pdf')
    parser.add_argument('--root_path', '-p', help='Path to the dataset', default='./Data/')

    args = parser.parse_args()

    subdirs = list()
    for d in os.listdir(args.root_path):
        if os.path.isdir(os.path.join(args.root_path, d)):
            subdirs.append(d)
    print('Data samples: ', subdirs)
    pbar = tqdm(subdirs, "Processing")
    for subdir in pbar:
        subpath = os.path.join(args.root_path, subdir)
        img_name = ''
        svg_name = ''
        dir_files = os.listdir(subpath)
        for f in dir_files:
            ending = f.split('.')[-1]

            if ending == 'pdf' and 'farba' in f:
                svg_name = f.replace('pdf', 'svg')
                if not os.path.exists(os.path.join(subpath, svg_name)):
                    pbar.set_description(f'Generating: {svg_name} ...')
                    os.system(f'inkscape --export-filename=\'{os.path.join(subpath, svg_name)}\' \'{os.path.join(subpath, f)}\'')
                else:
                    pbar.set_description(f'Found: {svg_name} file')
            elif ending == 'pdf':
                img_name = f.replace('pdf', 'png')
                if not os.path.exists(os.path.join(subpath, img_name)):
                    pbar.set_description(f'Generating: {img_name} ...')
                    os.system(f'inkscape --export-filename=\'{os.path.join(subpath, img_name)}\' \'{os.path.join(subpath, f)}\' --export-background-opacity=1')
                else:
                    pbar.set_description(f'Found: {img_name} source pdf file')
            elif ending == 'svg' and 'farba' in f:
                svg_name = f
                img_name = f.replace('svg', 'png').replace('_farba', '')

        if svg_name == '':
            print(f'Sample: {subdir} - No annotation file - skip')
            pbar.set_description("Done")
            continue
        elif img_name == '':
            print(f'Sample: {subdir} - No source pdf file - skip')
            pbar.set_description("Done")
            continue

        parse_label(subpath, img_name, svg_name, pbar)
        pbar.set_description("Done")
