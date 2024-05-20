from scanner.vis.xmlb import Xml_builder

class Visualizer():
    def __init__(self, outfile: str, width: int, height: int):
        self.x = Xml_builder(outfile)
        self.x.add_tag("svg")
        self.x.add_attribute("version", "1.1")
        self.x.add_attribute("xmlns", "http://www.w3.org/2000/svg")
        self.x.add_attribute("xmlns:xlink", "http://www.w3.org/1999/xlink")
        self.x.add_attribute("width", str(width))
        self.x.add_attribute("height", str(height))

    def __del__(self):
        self.x.close_all()

    def add_image(self, file: str):  
        self.x.add_tag("image")
        self.x.add_attribute("href", file)
        self.x.close_tag()

    def draw_circle(self, cx: float, cy: float, r: float, color: str = "black"):
        self.x.add_tag("circle")
        self.x.add_attribute("cx", str(cx))
        self.x.add_attribute("cy", str(cy))
        self.x.add_attribute("r", str(r))
        self.x.add_attribute("fill", color)
        self.x.close_tag()

    def draw_line(self, x1: float, y1: float, x2: float, y2: float, width: float = 1, color: str = "black"):
        self.x.add_tag("line")
        self.x.add_attribute("x1", str(x1))
        self.x.add_attribute("y1", str(y1))
        self.x.add_attribute("x2", str(x2))
        self.x.add_attribute("y2", str(y2))
        self.x.add_attribute("stroke", color)
        self.x.add_attribute("stroke-width", str(width))
        self.x.close_tag()
    
    def draw_text(self, x: float, y: float, text: str, color: str, size: float, font_weight: str = ""):
        self.x.add_tag("text")
        self.x.add_attribute("x", str(x))
        self.x.add_attribute("y", str(y))
        self.x.add_attribute("fill", color)
        self.x.add_attribute("font-size", str(size))
        self.x.add_attribute("font-weight", font_weight)
        self.x.add_text(text)
        self.x.close_tag()



