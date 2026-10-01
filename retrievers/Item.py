class Item:
    def __init__(self, item_id: str, ita_short_desc: str, eng_short_desc: str, ita_long_desc: str = None, eng_long_desc: str = None):
        self.item_id = item_id
        self.ita_short_desc = ita_short_desc
        self.eng_short_desc = eng_short_desc
        self.ita_long_desc = ita_long_desc
        self.eng_long_desc = eng_long_desc
        
    def __repr__(self):
        return f"Item(id={self.item_id}, ita_short_desc={self.ita_short_desc}, eng_short_desc={self.eng_short_desc}, ita_long_desc={self.ita_long_desc}, eng_long_desc={self.eng_long_desc})"