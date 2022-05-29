class MyIterater:
    def __init__(self, max_cnt) -> None:
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()

        self.cnt += 1
        return self.cnt


obj = MyIterater(5)
for x in obj:
    print(x)