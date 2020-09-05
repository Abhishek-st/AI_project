import os

cmd = 'abc2midi gen.abc -o hello.mid && timidity hello.mid'

os.system(cmd) 
