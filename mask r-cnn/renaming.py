import os
import xml.etree.ElementTree as ET

# List of folders to process
folders = ['Fast', 'Five', 'Four', 'Hello', 'Imfine', 'One', 'Three', 'Two', 'Wrong', 'Yes']

# Loop over each folder in the folders list
for folder in folders:
    # Set the directory containing your images and XML files for the current folder
    base_dir = f"FSL/{folder}/"
    image_dir = os.path.join(base_dir, "images/")
    xml_dir = os.path.join(base_dir, "annotations/")

    # List all XML files in the directory
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    # Sort files if needed to maintain specific order
    xml_files.sort()

    for idx, xml_file in enumerate(xml_files, start=1):
        # Construct the new file name with padding for 5 digits
        new_base_name = f"{idx:05d}"  # Adjusts the numbering format to 00001, 00002, ...
        new_image_name = f"{new_base_name}.jpg"
        new_xml_name = f"{new_base_name}.xml"

        # Check if the renamed XML already exists
        if not os.path.exists(os.path.join(xml_dir, new_xml_name)):
            # Rename the image file if the renamed version does not exist
            image_file = xml_file.replace('.xml', '.jpg')
            if os.path.exists(os.path.join(image_dir, image_file)):
                os.rename(os.path.join(image_dir, image_file), os.path.join(image_dir, new_image_name))
            
            # Parse the XML file
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            # Update the filename in the XML
            for filename in root.iter('filename'):
                filename.text = new_image_name

            # Set the path to reflect the relative path from the annotations to the images
            for path in root.iter('path'):
                path.text = f"../images/{new_image_name}"  # Updated to use relative path

            # Save the updated XML
            tree.write(os.path.join(xml_dir, new_xml_name))

            # Optionally, delete the old XML file if you want to keep only the renamed versions
            if xml_file != new_xml_name:  # Check if it's not the same file to avoid deleting the updated one
                os.remove(os.path.join(xml_dir, xml_file))
        else:
            # If the renamed XML exists, just update the path
            tree = ET.parse(os.path.join(xml_dir, new_xml_name))
            root = tree.getroot()

            # Update the path in the XML to reflect the correct relative path
            for path in root.iter('path'):
                path.text = f"../images/{new_image_name}"

            # Save the updates to the same XML file
            tree.write(os.path.join(xml_dir, new_xml_name))

    print(f"Process completed for folder: {folder}.")
