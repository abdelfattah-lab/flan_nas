
#!/bin/bash

# List of files to process
files=(
    Amoeba_samp_eff.csv
    DARTS_fix-w-d_samp_eff.csv
    DARTS_lr-wd_samp_eff.csv
    DARTS_samp_eff.csv
    ENAS_fix-w-d_samp_eff.csv
    ENAS_samp_eff.csv
    NASNet_samp_eff.csv
    PNAS_fix-w-d_samp_eff.csv
    PNAS_samp_eff.csv
    nb101_samp_eff.csv
    nb201_samp_eff.csv
    tb101_samp_eff.csv
)

# Directory where the files are located
dir="./"

# Process each file
for file in "${files[@]}"; do
    # Check if the file exists
    if [[ -f "${dir}${file}" ]]; then
        # Create a temporary copy
        cp "${dir}${file}" "${dir}${file}_tmp"
        
        # Replace the original with the temporary copy
        mv "${dir}${file}_tmp" "${dir}${file}"
        echo "Processed ${file}"
    else
        echo "File ${file} not found!"
    fi
done

echo "All files processed."

