# self/tcli.t — CLI wrapper cho compiler T thuần T (self/tcomp.t).
#
# Cách dùng:
#   trun tcomp.tb prog.t prog.tb          # dịch một file (có import tự gộp)
#   trun tcomp.tb -b prog.t prog.tb       # dịch + gộp import
#   trun tcomp.tb -b self/tcli.t t2.tb    # tự biên dịch (fixed point)
from tcomp import main_cli
main_cli()
