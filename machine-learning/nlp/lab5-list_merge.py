#!/usr/bin/env python
some_list =["first_name", "last_name", "age", "occupation"]
some_tuple = ("John", "Holloway", 35, "carpenter")

result = {some_list[i]: some_tuple[i] for i in range(len(some_list))}

print(result)
