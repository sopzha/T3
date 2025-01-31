import re

input_file = "input.sql"
output_file = "21-fhnk-schema.sql"
# Define the prefix
prefix = "fhnk"
quoted_table_pattern = re.compile(r'"([^"]+)"')


# Function to add prefix to a table name
def add_prefix(match):
    table_name = match.group()
    return f"{prefix}.{table_name}"


def main():
    # Input and output file paths

    # Open the input and output files
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        outfile.write(f"CREATE SCHEMA {prefix};\n")
        for line in infile:
            # Add prefix to DROP TABLE IF EXISTS statement
            drop_table_match = re.match(r"^\s*DROP\s+TABLE\s+IF\s+EXISTS\s+(\w+);", line, re.IGNORECASE)
            if not drop_table_match:
                drop_table_match = re.match(r'DROP TABLE IF EXISTS "(.*?)";', line, re.IGNORECASE)
            if drop_table_match:
                modified_line = re.sub(r"(\w+);", add_prefix, line)

            # Add prefix to CREATE TABLE statement (with or without parentheses)
            create_table_match = re.match(r"^\s*CREATE\s+TABLE\s+(\w+)\s*(\()?", line, re.IGNORECASE)
            if create_table_match:
                table_name = create_table_match.group(1)
                optional_opening_parentheses = create_table_match.group(2) if create_table_match.group(2) else ""
                modified_line = f"CREATE TABLE {prefix}.{table_name} {optional_opening_parentheses}\n"
            if not create_table_match:
                create_table_match = re.match(r'CREATE TABLE "(.*?)"\s*\(', line, re.IGNORECASE)
            if create_table_match:
                modified_line = f'CREATE TABLE "{prefix}{create_table_match.group(1)}" (\n'

            # Handle quoted table names in CREATE TABLE and DROP TABLE IF EXISTS statements
            quoted_table_match = re.search(quoted_table_pattern, line)
            if quoted_table_match:
                table_name = quoted_table_match.group(1)
                modified_line = re.sub(f'"{table_name}"', f'{prefix}."{table_name}"', line)

            # If the line didn't match any of the above patterns, keep it as is
            if not (drop_table_match or create_table_match):
                modified_line = line

            # Write the modified line to the output file
            outfile.write(modified_line)

    print(f"Table names in '{input_file}' with prefix '{prefix}' have been written to '{output_file}'.")


if __name__ == "__main__":
    main()
