# self/tcli.t — CLI wrapper cho compiler T thuần T (self/tcomp.t).
#
# Bootstrap (một lần duy nhất, cần Python):
#   python vm/tcomp.py -b self/tcli.t tcomp.tb
# Sau đó KHÔNG còn Python:
#   trun tcomp.tb prog.t prog.tb          # dịch một file
#   trun tcomp.tb -b prog.t prog.tb       # dịch + gộp import
#   trun tcomp.tb -b self/tcli.t t2.tb    # tự biên dịch (fixed point)
from tcomp import main_cli
main_cli()
