import json

d = {"name": "Alice", "age": 30, "city": "New York"}
# 将Python对象转换为JSON字符串


json_str = json.dumps(d,ensure_ascii=False)
#ensure_ascii=False参数可以让输出的JSON字符串中包含非ASCII字符（如中文）时正常显示，而不是被转义成\uXXXX的形式。
print(json_str)  # 输出: {"name": "Alice", "age": 30, "city": "New York"}

#print(str(d))这种强转会导致引号无法对应

