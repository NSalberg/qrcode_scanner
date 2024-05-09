from xml.xmlb import xml_builder
if __name__ == "__main__":
    x = xml_builder("test2.svg")
    x.add_tag("svg")
    x.add_attribute("version", "1.1")
    x.add_attribute("xmlns", "http://www.w2.org/2000/svg")
    x.add_attribute("xmlns:xlink", "http://www.w3.org/1999/xlink")
    x.add_attribute("width", "100")
    x.add_attribute("height", "100")
    x.add_tag("image")
    x.add_attribute("href","test.gif")
    print(x.tag_stack)
    x.finish()
    