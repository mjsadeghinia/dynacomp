import os
import argparse
import openpyxl

# Function to extract name, category, and weeks for a given sample name
def get_name_category_week(row):
    name = row[0].value   # Name
    category = row[3].value  # Category
    weeks = row[4].value  # Weeks
    return name, category, weeks

# Function to get the output directory path based on name, category, and weeks
def get_output_directory(directory_path, sample_name, category, weeks):
    # x is the weeks value minus the last character
    x = weeks[:-1]
    
    # s is the sample name with the second to last character replaced by '_'
    s = sample_name[:-2] + "_" + sample_name[-1]
    
    # Check the category to determine the output directory
    if category == "Sham":
        # Build the output directory for Sham category
        outdir = os.path.join(directory_path, f'SHAM/{x}weeks/{s}')
    else:
        # Convert the category value to double and multiply by 100
        c = float(category.replace(',', '.')) * 100
        # Build the output directory for non-Sham category
        outdir = os.path.join(directory_path, f'AS/{x}weeks/{c:.0f}/{s}')
    
    return outdir

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
        help="The sample name or 'all' to process all samples",
    )
    
    parser.add_argument(
        "-o",
        "--outdir",
        default="/home/shared/dynacomp/00_data/CineData/",
        type=str,
        help="The full address to the output directory",
    )
    
    args = parser.parse_args(args)
    excel_file_path = args.excel
    sample_name = args.sample
    outdir = args.outdir
    
    # Load the Excel workbook
    workbook = openpyxl.load_workbook(excel_file_path)
    sheet = workbook.active
    
    # Process all rows if sample_name is "all"
    if sample_name.lower() == "all":
        for row in sheet.iter_rows(min_row=2, values_only=False):
            name, category, weeks = get_name_category_week(row)
            output_directory = get_output_directory(outdir, name, category, weeks)
            if output_directory:
                print(f"{name}: {output_directory}")
        return
    
    # Otherwise, process only the specific sample
    for row in sheet.iter_rows(min_row=2, values_only=False):
        name, category, weeks = get_name_category_week(row)
        
        # Check if the current row matches the sample name
        if name == sample_name:
            output_directory = get_output_directory(outdir, name, category, weeks)
            if output_directory:
                print(output_directory)
            return

    print(f"Sample {sample_name} not found.")

if __name__ == "__main__":
    main()
