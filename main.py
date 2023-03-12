import numpy as np
from cv2 import cv2
import click
import os

from tqdm import tqdm

from augmentation.ink_transform import Bleed, ContrastChange, StrikeThrough, WhiteLines, Lightning, InkDrops, \
    CameraFocus
from augmentation.merge_ink_paper import merge
from augmentation.pipeline import Pipeline


def augment(ink: np.ndarray, paper: np.ndarray, ink_pipeline: Pipeline, post_pipeline: Pipeline):
    ink = cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY)
    ink = cv2.cvtColor(ink, cv2.COLOR_GRAY2RGB)

    w, h, c = ink.shape
    paper = cv2.resize(paper, (h, w), interpolation=cv2.INTER_AREA)

    paper = paper / 255
    ink = ink / 255

    ink = ink_pipeline.apply(ink)
    result = merge(ink, paper)
    result = post_pipeline.apply(result)
    return result


@click.command()
@click.option("--ink_images_folder")
@click.option("--papers_folder")
@click.option("--output_folder")
def main(ink_images_folder, papers_folder, output_folder):
    ink_pipeline = Pipeline()\
        .add(Bleed(), 1)\
        .add(ContrastChange(), 1)\
        .add(WhiteLines(), 0.5)

    global_counter = 0
    for ink_image in tqdm(os.listdir(ink_images_folder)):
        ink = cv2.imread(os.path.join(ink_images_folder, ink_image))
        if ink is None:
            continue

        for paper_image in os.listdir(papers_folder):
            paper = cv2.imread(os.path.join(papers_folder, paper_image))
            if paper is None:
                continue

            post_pipeline = Pipeline() \
                .add(InkDrops(max_y=ink.shape[0], max_x=ink.shape[1], number=10), 0.3) \
                .add(Lightning(max_y=ink.shape[0], max_x=ink.shape[1], number=1), 1) \
                .add(StrikeThrough(), 0.5) \
                .add(CameraFocus(), 0.5)

            result = augment(ink, paper, ink_pipeline, post_pipeline)

            filename = os.path.join(output_folder, f"{global_counter}.png")
            cv2.imwrite(filename, result * 255)
            global_counter += 1


if __name__ == '__main__':
    main()