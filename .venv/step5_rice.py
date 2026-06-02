import numpy as np

def rice_encode(n, k):
    # 1. ZigZag encoding (makes negative numbers positive)
    if n >= 0:
        val = 2 * n
    else:
        val = 2 * abs(n) - 1
    
    # 2. Calculate Quotient and Remainder
    m = 2**k
    quotient = val // m
    remainder = val % m
    
    # 3. Create the bits
    # Unary part: 'quotient' ones followed by a zero
    unary = "1" * quotient + "0"
    # Binary part: 'remainder' in k bits
    binary = format(remainder, f'0{k}b')
    
    return unary + binary

# Test with some residuals from your previous output
test_numbers = [0, 2, -1, 15]
k_parameter = 2

print(f"Rice Coding (k={k_parameter}):")
print("-" * 30)
for num in test_numbers:
    bits = rice_encode(num, k_parameter)
    print(f"Number: {num:3} | Bits: {bits:<10} | Length: {len(bits)} bits")

print("\nCompare this to a standard 16-bit WAV sample.")
print("The number '0' only took 3 bits instead of 16!")