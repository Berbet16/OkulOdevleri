#181805028_BETÜL_BAŞAN
#171805019_BETÜL_BERNA_SOYLU

funcdict = {
  2.36: 0.85866,
  2.37: 0.86289,
  2.38: 0.86710,
  2.39: 0.87129,
}

h = 0.01

def centralDiff1(x,h):  #1.türev
    derivative = ((funcdict[round(x+h,2)])-(funcdict[round(x-h,2)])) / (2*h)
    return derivative

def centralDiff2(x,h): #2.türev
    derivative = (funcdict[round(x+h,2)] + funcdict[round(x-h,2)] - (2*funcdict[round(x,2)])) / (h*h)
    return derivative

def backwardDiff1(x,h):  #1.türev
    derivative = ((funcdict[round(x-h-h,2)]) - (4*funcdict[round(x-h,2)]) + (3*funcdict[round(x,2)])) / (2*h)
    return derivative

def backwardDiff2(x,h): #2.türev
    derivative = ((4*funcdict[round(x-h-h, 2)]) - (funcdict[round(x-h-h-h, 2)]) - (5*funcdict[round(x-h, 2)]) + (2*funcdict[round(x, 2)])) / (h*h)
    return derivative

def forwardDiff1(x,h): #1.türev

    derivative = ((4*funcdict[round(x+h, 2)]) - (3*funcdict[round(x, 2)]) - (funcdict[round(x+h+h, 2)])) / (2*h)
    return derivative

def forwardDiff2(x,h): #2.türev
    derivative = ((2*funcdict[round(x, 2)]) - (5*funcdict[round(x+h, 2)]) + (4*funcdict[round(x+h+h, 2)]) - (funcdict[round(x+h+h+h, 2)])) / (h*h)
    return derivative

print(round(centralDiff1(2.38,h),2))
print(round(backwardDiff1(2.39,h),2))
print(round(forwardDiff1(2.37,h),2))

print(round(centralDiff2(2.38,h),2))
print(round(backwardDiff2(2.39,h),2))
print(round(forwardDiff2(2.36,h),2))
