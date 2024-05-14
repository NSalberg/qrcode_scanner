class Xml_builder:
    def __init__(self, file: str):
        self.tag_stack = []
        self.state = "tag"
        self.f = open(file, "w")
    
    def __del__(self):
        self.f.close()

    def add_tag(self, tag_name: str):
        if self.state == "tag":
            self.f.write(f"<{tag_name} ")
            self.tag_stack.append(tag_name)
            self.state = "attribute"
        elif self.state == "attribute":
            self.f.write(f">\n<{tag_name}")
            self.tag_stack.append(tag_name)
        else:
            print(f"error: invalid build state for add_tag: {self.state}")

    def add_attribute(self, attribute: str, data: str):
        if self.state == "attribute":
            self.f.write(f" {attribute}=\"{data}\"")
        else:
            print(f"error: invalid build state for add_atrribute: {self.state}")

    def close_tag(self):
        if self.state == "attribute":
            self.f.write("/> \n")
            self.tag_stack.pop()
            self.state = "tag"
        elif self.state == "tag":
            self.f.write(f"</{self.tag_stack.pop()}> \n")
        
    def close_all(self):
        while self.tag_stack:
            self.close_tag()

