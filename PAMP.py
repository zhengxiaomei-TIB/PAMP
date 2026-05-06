from Hyphae import HyphaeAnalyzer
from Pellets import pellet_analysis
import argparse

def main():
    parser = argparse.ArgumentParser(description='Pellet processing')
    parser.add_argument('--input_folder', type=str, default='./data/input/', help='Input folder')
    parser.add_argument('--output_folder', type=str, default='./data/output/', help='Output folder')
    parser.add_argument('--Flat_field_img', type=str, default=None, help='Flat field image path (optional, for flat-field correction)')
    parser.add_argument('--pellet_scale', type=float, default=2.14774, help='Pellet scale factor (default 2.14774)')
    parser.add_argument('--hyphae_scale', type=float, default=0.3373, help='Hyphae scale factor (default 0.3373)')
    
    args = parser.parse_args()

    pellet_analysis(
        args.input_folder,
        args.output_folder,
        Flat_field_img_path=args.Flat_field_img,
        scale=args.pellet_scale
    )

    hyphae_analyzer = HyphaeAnalyzer()
    hyphae_analyzer.pixel_size = args.hyphae_scale
    hyphae_analyzer.process_multiple_directories(args.input_folder, ['Spores', 'Hyphae'], args.output_folder)

if __name__ == '__main__':
    main()