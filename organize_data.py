import os
import argparse
import openpyxl

# Function to extract name, category, and weeks for a given sample name
def get_name_category_week(row):
    name = row[0].value   # Name
    category = row[3].value  # Category
    weeks = row[4].value  # Weeks
    return name, category, weeks

# Function to loop over the rows and get the output directory path
def get_output_directory(excel_file_path, directory_path, sample_name):
    # Load the Excel workbook
    workbook = openpyxl.load_workbook(excel_file_path)
    sheet = workbook.active
    
    # Loop over the rows in the Excel sheet (excluding the header row)
    for row in sheet.iter_rows(min_row=2, values_only=False):
        # Extract the name, category, and weeks using the helper function
        name, category, weeks = get_name_category_week(row)
        
        # Check if the current row matches the sample name
        if name == sample_name:
            # If the category is 'Sham', build the output directory path
            if category == "Sham":
                # x is the weeks value minus the last character
                x = weeks[:-1]
                
                # s is the sample name with the second to last character replaced by '_'
                s = name[:-2] + "_" + name[-1]
                
                # Build the output directory path
                outdir = os.path.join(directory_path, f'SHAM/{x}weeks/{s}')
                return outdir

    # If the sample name is not found, return None or raise an error
    return None


def main(args=None) -> int:
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        "-e",
        "--excel",
        default="/home/shared/dynacomp/00_data/CineData/MasterSheet_Code.xlsx",
        type=str,
        help="The full address to the excel file",
    )
    
    parser.add_argument(
        "-s",
        "--sample",
        default="OP100.1",
        type=str,
        help="The full address to the excel file",
    )
    
    parser.add_argument(
        "-o",
        "--outdir",
        default="/home/shared/dynacomp/00_data/CineData/",
        type=str,
        help="The full address to the excel file",
    )
    
    args = parser.parse_args(args)
    excel_file_path = args.excel
    sample_name = args.sample
    outdir = args.outdir
    
    output_directory = get_output_directory(excel_file_path, outdir, sample_name)
    print(output_directory)
    
if __name__ == "__main__":
    main()

    
    
    