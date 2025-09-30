#Жадібний алгоритм
def find_coins_greedy(amount):
    coins = [50, 25, 10, 5, 2, 1] 
    result = {} 

    for coin in coins:
        count = amount // coin  
        if count > 0:
            result[coin] = count
            amount -= coin * count  

    return result

#Динамічне програмування
def find_min_coins(amount):
    coins = [50, 25, 10, 5, 2, 1]
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  

    last = [-1] * (amount + 1)

    for coin in coins:
        for i in range(coin, amount + 1):
            if dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                last[i] = coin

    result = {}
    i = amount
    while i > 0:
        coin = last[i]
        result[coin] = result.get(coin, 0) + 1
        i -= coin

    return result

#Тестування
test_amounts = [63, 113, 199]

for amount in test_amounts:
    print(f"\nСума: {amount} копійок")
    print("Жадібний алгоритм:", find_coins_greedy(amount))
    print("Динамічне програмування:", find_min_coins(amount))