import json
from google.cloud import storage

def _add_single_field(dict, line):
    tokens = line.split()
    newKey = tokens[0]
    newValue = tokens[1]
    for token in range(2, len(tokens)):
        newValue += ' ' + str(tokens[token])
    if newValue.endswith(','):
        newValue = newValue[:-1]
    if newValue.startswith('"'):
        newValue = newValue[1:]
    if newValue.endswith('"'):
        newValue = newValue[:-1]
    newKey = newKey.replace(' ', '_')
    newKey = newKey.replace('-', '_')
    dict[newKey] = newValue

def _add_multiple_lines_field(dict, line, line_iterator):
    tokens = line.split()
    newKey = tokens[0]
    i = 1
    while True:
        if i == len(tokens):
            raise NameError("ERROR ADDING MULTIPLE LINES FIELD", filename)
        if tokens[i].startswith('"'):
            break
        newKey += tokens[i]
        i += 1
    newKey = newKey.replace(' ', '_')
    newKey = newKey.replace('-', '_')
    newValue = tokens[i]
    for j in range(i+1, len(tokens)):
        newValue += ' ' + tokens[j]
    # We assume that the value is inside " "
    newValue = newValue[1:]
    count = line.count('"')
    while True:
        if line.endswith('"') and (count%2)==0:
            newValue = newValue[:-1]
            dict[newKey] = newValue
            return
        if line.endswith('",') and (count%2)==0:
            newValue = newValue[:-2]
            dict[newKey] = newValue
            return
        line = next(line_iterator)
        count += line.count('"')
        newValue += line

def _add_list(line_iterator):
    newList = []
    while True:
        line = next(line_iterator)
        if line.endswith('},') or line.endswith('}'):
            return newList
        while (line.count('"')%2) != 0:
            line += next(line_iterator)
        # Delete spaces, " " and ,
        line = line.strip()[1:-1]
        if line[-1] == '"':
            line = line[:-1]
        newList.append(line)

def _add_simple_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        if line.endswith('},') or line.endswith('}'):
            return newDict
        _add_single_field(newDict, line)

def _add_multiple_lines_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        #if line.endswith('{'):
        #    continue
        if line.endswith('},') or line.endswith('}') and len(stripped_line)<=2:
            return newDict
        if '"' in line:
            _add_multiple_lines_field(newDict, line, line_iterator)
        else:
            _add_single_field(newDict, line)

def _add_list_of_dicts(line_iterator):
    newList = []
    while True:
        line = next(line_iterator)
        if line.endswith('{'):
            newList.append(_add_multiple_lines_dict(line_iterator))
        else:
            return newList

def _add_title_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        #if line.endswith('{'):
        #    continue
        if line.endswith('},') and len(stripped_line)<=2:
            return newDict
        if stripped_line.startswith(("name", "iso-jta", "ml-jta", "trans")):
            _add_multiple_lines_field(newDict, line, line_iterator)
        else:
            print(line)
            raise NameError("NEW FIELD AT TITLE LEVEL", filename)

def _add_authors_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if line.endswith('},') and len(stripped_line)<=2:
            return newDict
        if stripped_line.startswith("names ml"):
            newDict["names_ml"] = []
            while True:
                line = next(line_iterator)
                if line.endswith('}'):
                    break
                newDict["names_ml"].append(line.strip().replace(',', '').replace('"', ''))
        elif stripped_line.startswith("names std"):
            newDict["names_std"] = []
            while True:
                line = next(line_iterator)
                if line.endswith('{'):
                    newDict["names_std"].append(_add_multiple_lines_dict(line_iterator))
                if line.endswith('}'):
                    break
        else:
            print(newDict)
            raise NameError("NEW FIELD AT AUTHORS LEVEL", filename)

def _add_history_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if line.endswith('}') or line.endswith('},') and len(stripped_line)<=2:
            return newDict
        if stripped_line.startswith("pubstatus"):
            _add_single_field(newDict, line)
        elif stripped_line.startswith("date std"):
            newDict["date_std"] = _add_simple_dict(line_iterator)
        else:
            raise NameError("NEW FIELD AT HISTORY LEVEL", filename)

def _add_history_list(line_iterator):
    newList = []
    while True:
        line = next( line_iterator)
        if line.endswith('}'):
            return newList
        newList.append(_add_history_dict(line_iterator))

def _add_imp_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if line.endswith('},') or line.endswith('}') and len(stripped_line)<=2:
            return newDict
        if stripped_line.startswith("date std"):
            newDict["date_std"] = _add_simple_dict(line_iterator)
        elif stripped_line.startswith("pubstatus"):
            _add_single_field(newDict, line)
        elif stripped_line.startswith(("volume", "issue", "pages", "language")):
            _add_multiple_lines_field(newDict, line, line_iterator)
        elif stripped_line.startswith("history"):
            newDict["history"] = _add_history_list(line_iterator)
        elif stripped_line.startswith("retract"):
            newDict["retract"] = _add_multiple_lines_dict(line_iterator)
        else:
            raise NameError("NEW FIELD AT IMP LEVEL", filename)

def _add_fromjournal_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if line.endswith('},') and len(stripped_line) <=2:
            return newDict
        if stripped_line.startswith("title"):
            newDict["title"] = {}
            while True:
                line = next(line_iterator)
                if line.endswith('{'):
                    continue
                if line.endswith('}') or line.endswith('},'):
                    break
                _add_multiple_lines_field(newDict["title"], line, line_iterator)
        elif stripped_line.startswith("coll"):
            newDict["coll"] = {}
            while True:
                line = next(line_iterator)
                if line.endswith('{'):
                    continue
                if line.endswith('}') or line.endswith('},'):
                    break
                _add_multiple_lines_field(newDict["coll"], line, line_iterator)
        elif stripped_line.startswith("authors"):
            newDict["authors"] = _add_authors_dict(line_iterator)
        elif stripped_line.startswith("imp"):
            newDict["imp"] = _add_imp_dict(line_iterator)
        else:
            raise NameError("NEW FIELD AT FROM JOURNAL LEVEL", filename)

def _add_ids_dict(line_iterator):
    newDict = {}
    newDict["other"] = []
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if line.endswith('}') and len(stripped_line)<=2:
            return newDict
        if stripped_line.startswith(("pubmed","pmcid")):
            _add_single_field(newDict, line)
        elif stripped_line.startswith(("doi", "pii", "pmpid")):
            _add_multiple_lines_field(newDict, line, line_iterator)
        elif stripped_line.startswith("other"):
            otherDict = {}
            while True:
                line = next(line_iterator)
                if line.endswith('{'):
                    continue
                if line.endswith('}') or line.endswith('},'):
                    newDict["other"].append(otherDict)
                    break
                _add_multiple_lines_field(otherDict, line, line_iterator)
        else:
            raise NameError("NEW FIELD AT IDS LEVEL", filename)
    return

def _add_cit_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if line.endswith('},') and len(stripped_line)<=2:
            return newDict
        if stripped_line.startswith("title"):
            newDict["title"] = _add_title_dict(line_iterator)
        elif stripped_line.startswith("authors"):
            newDict["authors"] = _add_authors_dict(line_iterator)
        elif stripped_line.startswith("from journal"):
            newDict["from_journal"] = _add_fromjournal_dict(line_iterator)
        elif stripped_line.startswith("from book"):
            newDict["from_book"] = _add_fromjournal_dict(line_iterator)
        elif stripped_line.startswith("ids"):
            newDict["ids"] = _add_ids_dict(line_iterator)
        else:
            print(newDict)
            raise NameError("NEW FIELD AT CIT LEVEL", filename)

def _add_qual_dicts(line_iterator):
    newList = []
    while True:
        line = next(line_iterator)
        if line.endswith('{'):
            newList.append( _add_multiple_lines_dict(line_iterator))
        elif line.endswith('}'):
            return newList
        else:
            raise NameError("Error in qual", filename)

def _add_mesh_list(line_iterator):
    newList = []
    while True:
        line = next(line_iterator)
        if line.endswith(('}', '},')):
            return newList
        elif line.endswith('{'):
            newDict = {}
            while True:
                line = next(line_iterator)
                stripped_line = line.strip()
                if (line.endswith('},') or line.endswith('}')) and len(stripped_line)<=2:
                    newList.append(newDict)
                    break
                if stripped_line.startswith(("term", "mp")):
                    #newDict["term"] =_add_single_field(newDict, line)
                    _add_single_field(newDict, line)
                elif stripped_line.startswith("qual"):
                    newDict["qual"] = _add_qual_dicts(line_iterator)
                    #newDict["qual"] = _add_multiple_lines_dict(line_iterator)

def _add_substance_list(line_iterator):
    newList = []
    while True:
        line = next(line_iterator)
        if line.endswith('{'):
            newList.append(_add_multiple_lines_dict(line_iterator))
        elif line.endswith(('}', '},')):
            return newList

def _add_medent_dict(line_iterator):
    newDict = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if (line.endswith('}') or line.endswith('},')) and len(stripped_line)<=2:
            return newDict
        if stripped_line.startswith("em std"):
            newDict["em_std"] = _add_simple_dict(line_iterator)
        elif stripped_line.startswith("cit"):
            newDict["cit"] = _add_cit_dict(line_iterator)
        elif stripped_line.startswith("abstract"):
            _add_multiple_lines_field(newDict, line, line_iterator)
        elif stripped_line.startswith("mesh"):
            newDict["mesh"] = _add_mesh_list(line_iterator)
        elif stripped_line.startswith("substance"):
            newDict["substance"] = _add_substance_list(line_iterator)
        elif stripped_line.startswith("xref"):
            #newDict["xref"] = _add_list(line_iterator)
            newDict["xref"] = _add_list_of_dicts(line_iterator)
        elif stripped_line.startswith("idnum"):
            newDict["idnum"] = _add_list(line_iterator)
        elif stripped_line.startswith("gene"):
            newDict["gene"] = _add_list(line_iterator)
        elif stripped_line.startswith("pmid"):
            _add_single_field(newDict, line)
        elif stripped_line.startswith("pub-type"):
            newDict["pub_type"] = _add_list(line_iterator)
        elif stripped_line.startswith("status"):
            _add_single_field(newDict, line)
        else:
            print(newDict)
            raise NameError("NEW FIELD AT MEDENT LEVEL", filename)

def _read_pubmed_entry(line_iterator):
    entry = {}
    while True:
        line = next(line_iterator)
        stripped_line = line.strip()
        if line.endswith('}') and len(stripped_line)<=2:
            return entry
        if stripped_line.startswith("pmid"):
            _add_single_field(entry, line)
        elif stripped_line.startswith("medent"):
            entry["medent"] = _add_medent_dict(line_iterator)
        else:
            print(entry)
            raise NameError("NEW FIELD AT PUBMED LEVEL")

def parse(file_path, from_gc=False, input_bucket=None):
    """
    Retrieve the information encoded in an asn1 file with pubmed information.
    ...
    Parameters
    ----------
    file_path: str
        The path of the file the information is to be extracted from. It has to
        be a txt file with asn1 format containing pubmed data.
    from_gc: bool
        True if the input file path is a Google Cloud Bucket. False otherwise.
        Default is False.
    ...
    Returns
    -------
    A python list. Its elements are dictionaries, and each dictionary contains
    the information of a pubmed entry.
    """
    if from_gc:
        client = storage.Client()
        bucket = client.get_bucket(input_bucket)
        blob = bucket.get_blob(file_path)
        file = blob.download_as_string().decode("utf-8")
    else:
        file = ''
        with open(file_path, 'r') as fin:
            file += fin.read()
            file += '\n'

    entries = [] # The pubmed entries. List of dicts

    splitFile = file.splitlines()
    line_iterator = iter(splitFile)
    while True:
        try:
            line = next(line_iterator)
        except StopIteration:
            break

        if line == '':
            continue
        if line.startswith("Pubmed-entry ::="):
            entries.append(_read_pubmed_entry(line_iterator))

        #print(entries[-1]["pmid"])
    return entries

def _get_next_pmid(line_iterator):
    """
    Given a line iterator that iterates over lines in an asn1 pubmed file, this
    function returns the pmid of the next line.
    """
    line = next(line_iterator)
    pmid = line.split('pmid')[1].strip()[:-1]
    return int(pmid)

def _get_next_abstract(line_iterator):
    """
    Given a line iterator that iterates over lines in an asn1 pubmed file, this
    function returns the abstract of the pubmed entry we're currently parsing.
    If the entry has no abstract, it returns an empty string.
    """
    while True:
        line = next(line_iterator)
        if line.startswith('    abstract '):
            break
        if line.startswith('}'): # There is no abstract
            return ''
    abstract = line.split('abstract "')[1]
    while True:
        line = next(line_iterator)
        if line[1] == ' ':
            break
        abstract += line
    abstract = abstract[:-2]
    return abstract

def get_abstracts(file_path):
    """
    Get the abstracts from an asn1 file with pubmed information.
    ...
    Parameters
    ----------
    file_path: str
        The path of the file the abstracts are to be extracted from. It has to
        be a txt file with asn1 format containing pubmed data.
    ...
    Returns
    -------
    A python dictionary. Its keys are the pmids (int) of the pubmed entries
    present in the file, and its values are the corresponding abstracts (str).
    """
    file = ''
    with open(file_path, 'r') as fin:
        file += fin.read()
        file += '\n'

    splitFile = file.splitlines()
    line_iterator = iter(splitFile)

    abstracts = {}

    while True:
        try:
            line = next(line_iterator)
        except StopIteration:
            break

        if line == '':
            continue
        if line.startswith("Pubmed-entry ::="):
            pmid = _get_next_pmid(line_iterator)
            abstracts[pmid] = _get_next_abstract(line_iterator)
    return abstracts


def _one_line_json(input_json_string):
    output = ''
    lines = input_json_string.splitlines()
    for line in lines:
        if line.startswith(('[', ']')):
            continue
        if line.startswith(' }'): # Assuming indent is 1
            output += '}'
            output += '\n'
        else:
            output += line.strip()
    return output

def to_json(input_file_path, output_file_path=None, from_gc=False, input_bucket=None):
    """
    Retrieve the information encoded in an asn1 file with pubmed information,
    translate it to json format and store the output in a json file.
    ...
    Parameters
    ----------
    input_file_path: str
        The path of the file the information is to be extracted from. It has to
        be a txt file with asn1 format containing pubmed data.
    output_file_path: str
        The path for the output json file.
    from_gc: bool
        True if the input file path is a Google Cloud Bucket. False otherwise.
        Default is False.
    """
    global filename
    filename = input_file_path

    data = parse(input_file_path, from_gc=from_gc, input_bucket=input_bucket)

    if output_file_path==None:
        #return json.dumps(data, indent=4)
        try:
            output = _one_line_json(json.dumps(data, indent=1))
        except:
            print("Error in file", input_file_path)
            output = ''
        return output

    else:
        with open(output_file_path, 'w') as fout:
            json_data = json.dump(data, fout, indent=1)

"""
if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    return parse(file_path)
"""
