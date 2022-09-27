
import json
# file = open("stakedict.txt", "r")
# lines = file.readlines()
#
# for line in lines:
#     print(line)
#
#
#
# file1 = open("stakedict.txt", "r")
# lines1 = file1.readlines()
#
#
# for line in lines1:
#     print(line)


with open("stakedict.txt") as f:
    data = f.read()

print(type(data))
json_data = json.loads(data)
print(type(json_data))

# print(json_data)