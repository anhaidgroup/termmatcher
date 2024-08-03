class Node:
    def __init__(self):
        self.children = [None] * 26
        self.is_token = False
        self.token = None


def char_to_index(letter):
    return ord(letter) - ord("a")


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, token):
        curr = self.root
        for t in token:
            index = char_to_index(t)
            if not curr.children[index]:
                curr.children[index] = Node()

            curr = curr.children[index]

        curr.is_token = True
        curr.token = token
