import matplotlib.pyplot as plt

var = {
    "comm_round": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                   46, 47, 48, 49, 50],
    "stake_list": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 3, 0, 3, 30, 0, 3, 3, 30, 30],
                   [3, 3, 0, 3, 30, 60, 0, 30, 30, 3],
                   [60, 30, 30, 0, 39, 3, 3, 3, 2, 3],
                   [3, 41, 30, 39, 3, 0, 13, 3, 60, 3],
                   [39, 60, 17, 3, 30, 3, 16, 13, 54, 3],
                   [16, 54, 17, 30, 79, 13, 39, 6, 3, 3],
                   [79, 39, 6, 16, 17, 13, 16, 67, 43, 3],
                   [13, 43, 16, 52, 79, 16, 17, 3, 6, 80],
                   [16, 79, 26, 43, 27, 53, 52, 6, 13, 80],
                   [16, 26, 6, 52, 53, 26, 27, 92, 43, 93],
                   [26, 104, 37, 103, 52, 27, 43, 16, 6, 53],
                   [107, 37, 27, 10, 16, 52, 108, 47, 26, 57],
                   [107, 108, 26, 10, 47, 27, 64, 23, 52, 44], [47, 10, 23, 26, 64, 129, 52, 44, 107, 27],
                   [40, 39, 129, 52, 47, 23, 64, 44, 23, 107], [52, 47, 64, 23, 44, 52, 23, 129, 40, 107],
                   [64, 107, 23, 23, 64, 52, 52, 129, 40, 61],
                   [52, 114, 136, 23, 52, 23, 64, 40, 64, 68], [23, 23, 64, 127, 68, 136, 40, 65, 77, 52],
                   [64, 23, 136, 90, 68, 127, 36, 65, 52, 53],
                   [58, 59, 90, 65, 68, 64, 42, 127, 29, 136], [42, 58, 65, 42, 68, 140, 136, 90, 72, 64],
                   [90, 136, 68, 72, 42, 77, 42, 140, 65, 64],
                   [51, 86, 64, 68, 145, 74, 90, 72, 42, 140], [74, 90, 154, 95, 42, 81, 51, 140, 77, 64],
                   [74, 77, 140, 48, 64, 154, 57, 96, 95, 87],
                   [74, 84, 103, 161, 64, 48, 95, 140, 71, 87], [71, 140, 104, 103, 74, 95, 101, 161, 48, 64],
                   [48, 161, 64, 103, 74, 71, 104, 156, 117, 95],
                   [103, 170, 71, 104, 113, 126, 156, 74, 64, 48], [139, 113, 156, 103, 104, 183, 48, 77, 74, 71],
                   [103, 139, 77, 156, 71, 74, 104, 183, 126, 61],
                   [77, 156, 156, 143, 74, 103, 61, 104, 71, 183], [91, 156, 61, 143, 183, 103, 104, 77, 71, 173],
                   [103, 183, 173, 61, 77, 143, 190, 91, 71, 104],
                   [103, 173, 203, 104, 196, 61, 156, 91, 71, 77], [71, 87, 183, 103, 104, 156, 203, 81, 91, 196],
                   [93, 97, 71, 81, 183, 110, 202, 103, 156, 203],
                   [113, 203, 156, 103, 202, 81, 110, 93, 71, 199], [113, 156, 119, 199, 71, 112, 202, 93, 212, 90],
                   [113, 202, 212, 119, 169, 90, 112, 71, 106, 212],
                   [112, 169, 228, 212, 90, 119, 218, 113, 71, 106], [93, 109, 221, 172, 122, 116, 112, 228, 212, 74],
                   [93, 221, 116, 74, 112, 125, 122, 212, 172, 244],
                   [253, 93, 181, 122, 212, 112, 134, 230, 74, 116], [122, 144, 126, 112, 74, 253, 230, 103, 222, 181],
                   [253, 122, 103, 157, 194, 125, 126, 230, 222, 74],
                   [239, 103, 222, 135, 125, 131, 166, 194, 253, 74],
                   [239, 130, 166, 222, 108, 131, 253, 79, 140, 199]],
    "average": [0.0, 10.2, 16.2, 17.3, 19.5, 23.8, 26.0, 29.9, 32.5, 39.5, 43.4, 46.7, 48.7, 50.8, 52.9, 56.8, 58.1,
                61.5, 63.6, 67.5, 71.4, 73.8, 77.7, 79.6,
                83.2, 86.8, 89.2, 92.7, 96.1, 99.3, 102.9, 106.8, 109.4, 112.8, 116.2, 119.6, 123.5, 127.5, 129.9,
                133.1, 136.7, 140.6, 143.8, 145.9, 149.1, 152.7,
                156.7, 160.6, 164.2, 166.7]
}


plt.xlabel("Communication Rounds")
plt.ylabel("Stake Value")
# plt.title("Stake Accumulation Over the Time")
# for i in range(len(var["stake_list"][0])):
#     plt.plot(var["comm_round"],[pt[i] for pt in var["stake_list"]],label = 'Device %s'%i)

plt.title("Average Stake Value")

for i in range(len(var["stake_list"][0])):
    plt.plot(var["comm_round"], [pt[i] for pt in var["stake_list"]], label='Device %s' % i)
    plt.plot(var["comm_round"],var["average"],label = 'Device %s'%i)
# plt.legend()
plt.show()