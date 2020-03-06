import random
nums = [random.randint(1,100) for _ in range(20)]
print("Numbers: ", nums)
print("Average: ", sum(nums)/len(nums))
print("Even Nums: ", sum([1 for x in nums if not x % 2]))
print("1st Min/Max: ", min(nums), "/", max(nums))
print("2nd Min/Max: ", sorted(nums)[1], "/", sorted(nums, reverse = True)[1])