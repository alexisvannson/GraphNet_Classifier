import pandas as pd
from tqdm import tqdm
from gen_images_curve import generate_2Dcurve_image, generate_3dim_image, create_combined_3d_plot

def generate_excel(param_list, path='params.xlsx'):
    df = pd.DataFrame(param_list, columns=['a', 'b', 'c'])
    df.to_excel(path, index=False)

def generate_all_images_from_excel(excel_path='params.xlsx'):
    df = pd.read_excel(excel_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        generate_2Dcurve_image(row['a'], row['b'], row['c'], idx)

def generate_new_images_from_excel(excel_path='params.xlsx'):
    df = pd.read_excel(excel_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        create_combined_3d_plot(row['a'], row['b'], row['c'], idx)