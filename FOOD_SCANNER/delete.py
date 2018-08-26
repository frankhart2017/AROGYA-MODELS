num = [42,52,71,73,90,96,101,103,107,110,112,130,132,134,138,144,156, 158, 160,163,168,169,171,177,185,188,193,194,200,223,227,229,238,240,243,244,248,260,264,266,270,271,
275,277,287,288,292,296,298,300,319,320,321,332,341,360,361,371,374,376,377,381,386,391,394,398]

import os
from pathlib import Path
name = 'dal makhni_'
for n in num:
	filen = Path(name + str(n) + ".jpg")
	if filen.exists():
		os.remove(filen)
	elif Path(name + str(n) + '.jpeg').exists():
		os.remove(Path(name+ str(n) + '.jpeg'))
	else:
		filen = Path(name + str(n) + ".png")
		os.remove(filen)
	print(n, "deleted!")
