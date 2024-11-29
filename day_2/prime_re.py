# regex for prime numbers

# source: https://illya.sh/the-codeumentary-blog/regular-expression-check-if-number-is-prime/
# source: https://www.youtube.com/watch?v=5vbk0TwkokM


import re



def is_prime(n):
    return not re.match(r'^.?$|^(..+?)\1+$', '1'*n)

for i in range(1, 20):
    print(i, is_prime(i))