class Constants:
    SEED = 100 # in hope of achieving reproducible results
    DATA_DIR = '../../web_embedding' # contains imgs/*.png, bboxes/*.pkl, and additional_features/*.pkl (optional)
    SPLIT_DIR = '../../web_embedding/splits' # contains Fold-%d dir containing {train|val|test}_{imgs|domains}.txt and webpage_info.csv (optional)
    CLASS_NAMES = ['Price', 'Name', 'Image'] # Accuracies of class-0 (BG) are ignored
    N_CLASSES = 4 # CLASS_NAMES + BG
    IMG_HEIGHT = 1280 # image assumed to have same height and width
    OUTPUT_DIR = 'results_5-Fold_CV' # results dir is created here
