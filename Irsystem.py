import re

# Documents
documents = {
    1: "Information retrieval helps users find relevant information from documents.",
    2: "Search engines use information retrieval techniques to find useful documents.",
    3: "Machine learning is used in many modern information retrieval systems.",
    4: "Python is a popular programming language for data analysis and machine learning.",
    5: "Natural language processing helps computers understand human language.",
    6: "Search engines index web pages and retrieve relevant results for users.",
    7: "Boolean retrieval uses operators such as AND OR and NOT.",
    8: "An inverted index maps terms to the documents in which they appear."
}

# Tokenize
def tokenize(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


# Dictionary
def create_dictionary(documents):
    dictionary = set()

    for text in documents.values():
        words = tokenize(text)
        dictionary.update(words)

    return sorted(dictionary)


# Index
def create_inverted_index(documents):
    inverted_index = {}

    for doc_id, text in documents.items():
        words = tokenize(text)

        for word in words:
            if word not in inverted_index:
                inverted_index[word] = set()

            inverted_index[word].add(doc_id)

    return inverted_index


# AND
def boolean_and(term1, term2, index):
    result1 = index.get(term1.lower(), set())
    result2 = index.get(term2.lower(), set())

    return result1 & result2


# OR
def boolean_or(term1, term2, index):
    result1 = index.get(term1.lower(), set())
    result2 = index.get(term2.lower(), set())

    return result1 | result2


# NOT
def boolean_not(term, index, documents):
    all_documents = set(documents.keys())
    result = index.get(term.lower(), set())

    return all_documents - result


# Display
def display_documents(documents):
    print("\nDOCUMENT COLLECTION")
    print("-" * 50)

    for doc_id, text in documents.items():
        print(f"Document {doc_id}: {text}")


# Main
def main():

    # Create
    dictionary = create_dictionary(documents)
    inverted_index = create_inverted_index(documents)

    # Display
    display_documents(documents)

    # Dictionary
    print("\n\nDICTIONARY")
    print("-" * 50)

    print(dictionary)
    print("\nTotal unique terms:", len(dictionary))

    # Index
    print("\n\nINVERTED INDEX")
    print("-" * 50)

    for term in sorted(inverted_index):
        print(f"{term}: {sorted(inverted_index[term])}")

    # Retrieval
    print("\n\nBOOLEAN RETRIEVAL")
    print("-" * 50)

    print("AND (information AND retrieval):")
    print(boolean_and("information", "retrieval", inverted_index))

    print("\nOR (python OR machine):")
    print(boolean_or("python", "machine", inverted_index))

    print("\nNOT (NOT python):")
    print(boolean_not("python", inverted_index, documents))


# Run
if __name__ == "__main__":
    main()