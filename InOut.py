import pickle as pkl
def create_file(file_path, file_content, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as file:
        file.write(file_content)

# This function creates a file with the given content in our format
def create_fileformatted(file_path, file_content, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as file:
        for sentence in file_content:
            file.write(str(sentence)+"\n")
def read_file_content(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()
CONSTANTS_PATH='Const'
def read_file_frompickle(file_path, picklename):
    with open(file_path + picklename, 'rb') as file:
        returnlist = pkl.load(file)
    return returnlist
def write_Pickle(file_path,filename, pickle_list):
    with open(file_path + filename, 'wb') as file:
        pkl.dump(pickle_list, file)
