import sys, importlib, pprint
print(sys.path)
try:
    import mathx
    print('Imported', mathx)
except Exception as e:
    print('Error', e)
