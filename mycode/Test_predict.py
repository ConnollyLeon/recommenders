f = open('prediction.txt')
lines = f.readlines()
print(len(lines))
print(lines[-1])
for line in lines:
    u, l = line.strip().split()
    l =[int(x) for x in l.strip('[]').split(',')]
    assert max(l)==len(l)
