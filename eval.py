import math
import os
import sys
import subprocess
import time

vels = {
    "HCFB": [math.inf, -math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, -math.inf, math.inf, math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf, -math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, -math.inf, math.inf, -math.inf, -math.inf, math.inf],
    "HCP": [1.621739044286087, 0.5120315517529808, 1.600031953540813, 1.3230884164208945, 0.818337879736718, 1.8390578197915646, 0.875129453163364, 1.2301006596606492, 1.9017166617863595, 0.17889633184676712, 1.9932221030952089, 1.3484648368390593, 1.1518208894213369, 0.8479821863457024, 1.1127970111023426, 0.03588457187991345, 1.525161566415382, 0.03413930365809237, 0.6481791204556626, 1.5703290150877156, 0.07574701436183395, 0.9473944477019758, 0.7762348890093063, 0.3303059755334363, 1.7210104128224775, 1.285390167914429, 1.4277798133230495, 1.4349238383912646, 0.4889460701412387, 1.2918231586056708, 0.5140048289371433, 1.530888589125165, 0.38416948358415404, 0.7301007759699705, 0.5872963834433034, 0.8579206201854694, 1.7464146092825317, 0.275386527756438, 0.00211481676132097, 1.2958350976765327],
    "HCF": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCF50": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCF30": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "AFB": [math.inf, -math.inf, math.inf, math.inf, math.inf, math.inf, -math.inf, math.inf, -math.inf, -math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf, math.inf, math.inf, math.inf, -math.inf, math.inf, math.inf, -math.inf, math.inf, math.inf, -math.inf, -math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf, math.inf, -math.inf, -math.inf, -math.inf, -math.inf, math.inf],
    "AP": [0.7053388502956301, 1.2833862787089338, 2.5260050697968115, 0.8974870166302051, 0.007531547469744382, 0.5096374564521104, 1.2187510522505263, 0.1049886922212474, 2.8689360338669365, 1.6384676027462461, 1.0387508895658106, 0.6879434676873194, 0.6421632092236444, 0.42283264082528504, 1.4857185257156393, 1.7435905569754317, 0.6066977325531295, 2.41161165326549, 0.2240416269755785, 2.3424147557021655, 1.5577322738483514, 2.5838131067460592, 1.9558139919057878, 2.9681206206809856, 1.1468361122606208, 1.0972125353243145, 1.1954566572207752, 0.36205746198951505, 0.11832704596361898, 0.049266694609681916, 0.7028020664823245, 0.5301775734326277, 0.24798822606050064, 1.3770351847281042, 0.7436159396464881, 2.188620026743158, 1.8178631246175083, 2.131074514392589, 2.84933325663815, 1.9460722312800192],
    "AF": [-2.5571114036000893, 2.370787529734624, -1.4722178997163375, -2.5060616662640696, -2.15860358396866, -1.8334454911504532, 0.04597115185150491, 1.6251530033017243, 2.7651907046359216, 1.2524130477638717, -0.753161638680544, 0.7238729656052252, 2.5548894952603343, 1.033855607236501, 2.085872724039568, -2.45372204478803, -2.9364707418170983, 0.34690715136410466, 1.5405296236214934, 2.146126113652322, 2.995544828506352, -1.2956401167718177, 0.060096848894452215, 1.9237446411190184, -1.0366421749320838, 2.9982435751744125, 1.315294892626575, 2.85874691886284, 1.9564084836506384, -1.4542237448242639, 0.23280201896681785, 0.022090167676853234, -2.153485268123983, 0.7096707921605585, -1.000441722979282, 2.8945882455245524, -1.4754743481572015, -0.2568700734270717, -0.5468445332186187, -2.143314994126202],
    "AD": [[2.5907082022843033, -2.652657185659601], [0.15370897035836872, -0.7384922094295558], [-0.664652809087495, 1.4550838491737377], [-1.7465970054440583, -1.4002569578179804], [-1.2870063270928112, -1.8160625591113844], [0.00797936072637162, 1.3322509439932686], [-2.0131102220680566, 1.0916863865888509], [-2.6252947439499867, -1.6080872405622544], [1.3937619528840512, 2.744601809520729], [2.811962472893856, 2.4229388908037635], [-0.8231603481116618, -0.8788486833102436], [-1.6452416045946434, -1.2364038427626258], [0.4072580169217459, 2.7747379513537815], [0.4544844010075675, 0.08853004257269381], [2.421073292498834, 0.6031077791318626], [-0.041263031582635, 2.5037760378504554], [-2.7850036992282865, 1.7278717220394473], [-2.254817924545314, 1.7866250677126772], [-1.0908902580884363, 2.8499580659459944], [-2.0826055915438495, 1.6728814875325417], [-0.6629190071751214, -2.4612088523248117], [1.3982413416735548, -0.9637436315569907], [-1.6128818686746977, -0.35345311828056936], [-0.8751063460652708, -1.0743390340569483], [-1.3861565343075952, 2.2593881170451917], [1.3710076884316411, 2.569301292321688], [1.256683004907929, 2.722144240582561], [-0.9957610711036318, -2.123385088703925], [-0.5286171246899598, -2.0537818823856284], [2.6683929940203504, 0.8014091683897284], [-2.894561587206404, -0.5040682460071433], [-2.727754722509557, -2.7495019877588884], [2.7215512841828042, -2.421790501025397], [-1.3571598077100135, -1.0770910909796791], [2.429332996578152, 0.206127379267671], [2.3566340009428286, -2.4955082668637614], [-1.6953918267164312, -0.5997801393857882], [0.44439095257038996, 2.473882131022747], [-2.087934926391811, -2.825260666250113], [0.8056303457490634, -2.5860818912831167]],
    "AD30": [[2.5907082022843033, -2.652657185659601], [0.15370897035836872, -0.7384922094295558], [-0.664652809087495, 1.4550838491737377], [-1.7465970054440583, -1.4002569578179804], [-1.2870063270928112, -1.8160625591113844], [0.00797936072637162, 1.3322509439932686], [-2.0131102220680566, 1.0916863865888509], [-2.6252947439499867, -1.6080872405622544], [1.3937619528840512, 2.744601809520729], [2.811962472893856, 2.4229388908037635], [-0.8231603481116618, -0.8788486833102436], [-1.6452416045946434, -1.2364038427626258], [0.4072580169217459, 2.7747379513537815], [0.4544844010075675, 0.08853004257269381], [2.421073292498834, 0.6031077791318626], [-0.041263031582635, 2.5037760378504554], [-2.7850036992282865, 1.7278717220394473], [-2.254817924545314, 1.7866250677126772], [-1.0908902580884363, 2.8499580659459944], [-2.0826055915438495, 1.6728814875325417], [-0.6629190071751214, -2.4612088523248117], [1.3982413416735548, -0.9637436315569907], [-1.6128818686746977, -0.35345311828056936], [-0.8751063460652708, -1.0743390340569483], [-1.3861565343075952, 2.2593881170451917], [1.3710076884316411, 2.569301292321688], [1.256683004907929, 2.722144240582561], [-0.9957610711036318, -2.123385088703925], [-0.5286171246899598, -2.0537818823856284], [2.6683929940203504, 0.8014091683897284], [-2.894561587206404, -0.5040682460071433], [-2.727754722509557, -2.7495019877588884], [2.7215512841828042, -2.421790501025397], [-1.3571598077100135, -1.0770910909796791], [2.429332996578152, 0.206127379267671], [2.3566340009428286, -2.4955082668637614], [-1.6953918267164312, -0.5997801393857882], [0.44439095257038996, 2.473882131022747], [-2.087934926391811, -2.825260666250113], [0.8056303457490634, -2.5860818912831167]],
    "AD50": [[2.5907082022843033, -2.652657185659601], [0.15370897035836872, -0.7384922094295558], [-0.664652809087495, 1.4550838491737377], [-1.7465970054440583, -1.4002569578179804], [-1.2870063270928112, -1.8160625591113844], [0.00797936072637162, 1.3322509439932686], [-2.0131102220680566, 1.0916863865888509], [-2.6252947439499867, -1.6080872405622544], [1.3937619528840512, 2.744601809520729], [2.811962472893856, 2.4229388908037635], [-0.8231603481116618, -0.8788486833102436], [-1.6452416045946434, -1.2364038427626258], [0.4072580169217459, 2.7747379513537815], [0.4544844010075675, 0.08853004257269381], [2.421073292498834, 0.6031077791318626], [-0.041263031582635, 2.5037760378504554], [-2.7850036992282865, 1.7278717220394473], [-2.254817924545314, 1.7866250677126772], [-1.0908902580884363, 2.8499580659459944], [-2.0826055915438495, 1.6728814875325417], [-0.6629190071751214, -2.4612088523248117], [1.3982413416735548, -0.9637436315569907], [-1.6128818686746977, -0.35345311828056936], [-0.8751063460652708, -1.0743390340569483], [-1.3861565343075952, 2.2593881170451917], [1.3710076884316411, 2.569301292321688], [1.256683004907929, 2.722144240582561], [-0.9957610711036318, -2.123385088703925], [-0.5286171246899598, -2.0537818823856284], [2.6683929940203504, 0.8014091683897284], [-2.894561587206404, -0.5040682460071433], [-2.727754722509557, -2.7495019877588884], [2.7215512841828042, -2.421790501025397], [-1.3571598077100135, -1.0770910909796791], [2.429332996578152, 0.206127379267671], [2.3566340009428286, -2.4955082668637614], [-1.6953918267164312, -0.5997801393857882], [0.44439095257038996, 2.473882131022747], [-2.087934926391811, -2.825260666250113], [0.8056303457490634, -2.5860818912831167]],
    "ADDM30": [[2.5907082022843033, -2.652657185659601], [0.15370897035836872, -0.7384922094295558], [-0.664652809087495, 1.4550838491737377], [-1.7465970054440583, -1.4002569578179804], [-1.2870063270928112, -1.8160625591113844], [0.00797936072637162, 1.3322509439932686], [-2.0131102220680566, 1.0916863865888509], [-2.6252947439499867, -1.6080872405622544], [1.3937619528840512, 2.744601809520729], [2.811962472893856, 2.4229388908037635], [-0.8231603481116618, -0.8788486833102436], [-1.6452416045946434, -1.2364038427626258], [0.4072580169217459, 2.7747379513537815], [0.4544844010075675, 0.08853004257269381], [2.421073292498834, 0.6031077791318626], [-0.041263031582635, 2.5037760378504554], [-2.7850036992282865, 1.7278717220394473], [-2.254817924545314, 1.7866250677126772], [-1.0908902580884363, 2.8499580659459944], [-2.0826055915438495, 1.6728814875325417], [-0.6629190071751214, -2.4612088523248117], [1.3982413416735548, -0.9637436315569907], [-1.6128818686746977, -0.35345311828056936], [-0.8751063460652708, -1.0743390340569483], [-1.3861565343075952, 2.2593881170451917], [1.3710076884316411, 2.569301292321688], [1.256683004907929, 2.722144240582561], [-0.9957610711036318, -2.123385088703925], [-0.5286171246899598, -2.0537818823856284], [2.6683929940203504, 0.8014091683897284], [-2.894561587206404, -0.5040682460071433], [-2.727754722509557, -2.7495019877588884], [2.7215512841828042, -2.421790501025397], [-1.3571598077100135, -1.0770910909796791], [2.429332996578152, 0.206127379267671], [2.3566340009428286, -2.4955082668637614], [-1.6953918267164312, -0.5997801393857882], [0.44439095257038996, 2.473882131022747], [-2.087934926391811, -2.825260666250113], [0.8056303457490634, -2.5860818912831167]],
    "ADSSE30": [[2.5907082022843033, -2.652657185659601], [0.15370897035836872, -0.7384922094295558], [-0.664652809087495, 1.4550838491737377], [-1.7465970054440583, -1.4002569578179804], [-1.2870063270928112, -1.8160625591113844], [0.00797936072637162, 1.3322509439932686], [-2.0131102220680566, 1.0916863865888509], [-2.6252947439499867, -1.6080872405622544], [1.3937619528840512, 2.744601809520729], [2.811962472893856, 2.4229388908037635], [-0.8231603481116618, -0.8788486833102436], [-1.6452416045946434, -1.2364038427626258], [0.4072580169217459, 2.7747379513537815], [0.4544844010075675, 0.08853004257269381], [2.421073292498834, 0.6031077791318626], [-0.041263031582635, 2.5037760378504554], [-2.7850036992282865, 1.7278717220394473], [-2.254817924545314, 1.7866250677126772], [-1.0908902580884363, 2.8499580659459944], [-2.0826055915438495, 1.6728814875325417], [-0.6629190071751214, -2.4612088523248117], [1.3982413416735548, -0.9637436315569907], [-1.6128818686746977, -0.35345311828056936], [-0.8751063460652708, -1.0743390340569483], [-1.3861565343075952, 2.2593881170451917], [1.3710076884316411, 2.569301292321688], [1.256683004907929, 2.722144240582561], [-0.9957610711036318, -2.123385088703925], [-0.5286171246899598, -2.0537818823856284], [2.6683929940203504, 0.8014091683897284], [-2.894561587206404, -0.5040682460071433], [-2.727754722509557, -2.7495019877588884], [2.7215512841828042, -2.421790501025397], [-1.3571598077100135, -1.0770910909796791], [2.429332996578152, 0.206127379267671], [2.3566340009428286, -2.4955082668637614], [-1.6953918267164312, -0.5997801393857882], [0.44439095257038996, 2.473882131022747], [-2.087934926391811, -2.825260666250113], [0.8056303457490634, -2.5860818912831167]],
    "ADRSE30": [[2.5907082022843033, -2.652657185659601], [0.15370897035836872, -0.7384922094295558], [-0.664652809087495, 1.4550838491737377], [-1.7465970054440583, -1.4002569578179804], [-1.2870063270928112, -1.8160625591113844], [0.00797936072637162, 1.3322509439932686], [-2.0131102220680566, 1.0916863865888509], [-2.6252947439499867, -1.6080872405622544], [1.3937619528840512, 2.744601809520729], [2.811962472893856, 2.4229388908037635], [-0.8231603481116618, -0.8788486833102436], [-1.6452416045946434, -1.2364038427626258], [0.4072580169217459, 2.7747379513537815], [0.4544844010075675, 0.08853004257269381], [2.421073292498834, 0.6031077791318626], [-0.041263031582635, 2.5037760378504554], [-2.7850036992282865, 1.7278717220394473], [-2.254817924545314, 1.7866250677126772], [-1.0908902580884363, 2.8499580659459944], [-2.0826055915438495, 1.6728814875325417], [-0.6629190071751214, -2.4612088523248117], [1.3982413416735548, -0.9637436315569907], [-1.6128818686746977, -0.35345311828056936], [-0.8751063460652708, -1.0743390340569483], [-1.3861565343075952, 2.2593881170451917], [1.3710076884316411, 2.569301292321688], [1.256683004907929, 2.722144240582561], [-0.9957610711036318, -2.123385088703925], [-0.5286171246899598, -2.0537818823856284], [2.6683929940203504, 0.8014091683897284], [-2.894561587206404, -0.5040682460071433], [-2.727754722509557, -2.7495019877588884], [2.7215512841828042, -2.421790501025397], [-1.3571598077100135, -1.0770910909796791], [2.429332996578152, 0.206127379267671], [2.3566340009428286, -2.4955082668637614], [-1.6953918267164312, -0.5997801393857882], [0.44439095257038996, 2.473882131022747], [-2.087934926391811, -2.825260666250113], [0.8056303457490634, -2.5860818912831167]],
    "ADRRD30": [[2.5907082022843033, -2.652657185659601], [0.15370897035836872, -0.7384922094295558], [-0.664652809087495, 1.4550838491737377], [-1.7465970054440583, -1.4002569578179804], [-1.2870063270928112, -1.8160625591113844], [0.00797936072637162, 1.3322509439932686], [-2.0131102220680566, 1.0916863865888509], [-2.6252947439499867, -1.6080872405622544], [1.3937619528840512, 2.744601809520729], [2.811962472893856, 2.4229388908037635], [-0.8231603481116618, -0.8788486833102436], [-1.6452416045946434, -1.2364038427626258], [0.4072580169217459, 2.7747379513537815], [0.4544844010075675, 0.08853004257269381], [2.421073292498834, 0.6031077791318626], [-0.041263031582635, 2.5037760378504554], [-2.7850036992282865, 1.7278717220394473], [-2.254817924545314, 1.7866250677126772], [-1.0908902580884363, 2.8499580659459944], [-2.0826055915438495, 1.6728814875325417], [-0.6629190071751214, -2.4612088523248117], [1.3982413416735548, -0.9637436315569907], [-1.6128818686746977, -0.35345311828056936], [-0.8751063460652708, -1.0743390340569483], [-1.3861565343075952, 2.2593881170451917], [1.3710076884316411, 2.569301292321688], [1.256683004907929, 2.722144240582561], [-0.9957610711036318, -2.123385088703925], [-0.5286171246899598, -2.0537818823856284], [2.6683929940203504, 0.8014091683897284], [-2.894561587206404, -0.5040682460071433], [-2.727754722509557, -2.7495019877588884], [2.7215512841828042, -2.421790501025397], [-1.3571598077100135, -1.0770910909796791], [2.429332996578152, 0.206127379267671], [2.3566340009428286, -2.4955082668637614], [-1.6953918267164312, -0.5997801393857882], [0.44439095257038996, 2.473882131022747], [-2.087934926391811, -2.825260666250113], [0.8056303457490634, -2.5860818912831167]],
    "HUFB": [math.inf, -math.inf, math.inf, math.inf, math.inf, math.inf, -math.inf, math.inf, -math.inf, -math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf, math.inf, math.inf, math.inf, -math.inf, math.inf, math.inf, -math.inf, math.inf, math.inf, -math.inf, -math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf, math.inf, -math.inf, -math.inf, -math.inf, -math.inf, math.inf],
    "HCFDM30": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "DV30": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "SRV30": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCFSSE30": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCFRSE30": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCFRRD30": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCFDM50": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "DV50": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "SRV50": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCFSSE50": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCFRSE50": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HCFRRD50": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],
    "HC-flip": [-0.730113936377963, 0.11907929481668278, -1.3543698742411587, 0.6337570297496722, -0.6265919742911406, -1.5832607528269484, 1.5481589531538384, -0.6070265000766017, 1.3763055176186398, -0.7990751875854549, -1.847517167587816, 0.6184574629125303, -1.4345230506210447, -0.24706877364398627, 1.4136736284095415, 0.4819646574378744, 1.6787233305706692, -1.3809677381145713, 0.5852122192333478, 1.153482714819555, -0.0033608023136570964, 0.8805843388572985, -1.4800964630412303, -0.0716681927457512, 0.20130410011034172, -0.9528783573529007, 1.9798937278332853, -1.7698774766039693, 1.6377186691485885, -0.3651846477902274, -0.7149272451142736, -1.3041723292904819, -1.5288256848476705, 0.04031722270744842, -1.9153102342977815, 1.7656318748338888, 0.04044230773279889, -1.3287582530024422, 1.6698433382500721, 0.03344255635917914],

}


tasks = {
    "HCFB": "/tiger/u/lando/jobs/HCFB-eval.sh",
    "HCP": "/tiger/u/lando/jobs/HCP-eval.sh",
    "HCF": "/tiger/u/lando/jobs/HCF-eval.sh",
    "HCF50": "/tiger/u/lando/jobs/HCF50-eval.sh",
    "AFB": "/tiger/u/lando/jobs/AFB-eval.sh",    
    "AP": "/tiger/u/lando/jobs/AP-eval.sh",
    "AF": "/tiger/u/lando/jobs/AF-eval.sh",
    "AD": "/tiger/u/lando/jobs/AD-eval.sh",
    "AD30": "/tiger/u/lando/jobs/AD30-eval.sh",
    "AD50": "/tiger/u/lando/jobs/AD50-eval.sh",
    "HUFB": "/tiger/u/lando/jobs/HUFB-eval.sh",
    "HCFDM30": "/tiger/u/lando/jobs/DM30-eval.sh",
    "DV30": "/tiger/u/lando/jobs/DV30-eval.sh",
    "SRV30": "/tiger/u/lando/jobs/SRV30-eval.sh",
    "HCFSSE30": "/tiger/u/lando/jobs/SSE30-eval.sh",
    "HCFRSE30": "/tiger/u/lando/jobs/RSE30-eval.sh",
    "HCFRRD30": "/tiger/u/lando/jobs/RRD30-eval.sh",
    "HCFDM50": "/tiger/u/lando/jobs/DM50-eval.sh",
    "DV50": "/tiger/u/lando/jobs/DV50-eval.sh",
    "SRV50": "/tiger/u/lando/jobs/SRV50-eval.sh",
    "HCFSSE50": "/tiger/u/lando/jobs/SSE50-eval.sh",
    "HCFRSE50": "/tiger/u/lando/jobs/RSE50-eval.sh",
    "HCFRRD50": "/tiger/u/lando/jobs/RRD50-eval.sh",
    "HCF30": "/tiger/u/lando/jobs/HCF30-eval.sh",
    "ADDM30": "/tiger/u/lando/jobs/ADDM30-eval.sh",
    "ADSSE30": "/tiger/u/lando/jobs/ADSSE30-eval.sh",
    "ADRSE30": "/tiger/u/lando/jobs/ADRSE30-eval.sh",
    "ADRRD30": "/tiger/u/lando/jobs/ADRRD30-eval.sh",
     "HC-flip": "/tiger/u/lando/jobs/HC-flip-eval.sh",
}

def main(exp):
    if exp not in vels:
        raise Exception("Unkown eval experiment")

    jobs = []
    e = os.environ.copy()
    for vel in vels[exp]:
        if vel == math.inf:
            e["VELOCITY"] = "[.inf]"
        elif vel == -math.inf:
            e["VELOCITY"] = "[-.inf]"
        else:
            e["VELOCITY"] = f"[{vel}]".replace(" ", "")

        prev = {}
        if os.environ['AZURE'] != "YES":
            x = subprocess.run(args=["sbatch", tasks[exp], "-o", "/tiger/u/lando/jobs/slurm-%j.out"], env=e, stdout=subprocess.PIPE)
            job = int(x.stdout[-7:-1])
            jobs.append(job)
        else:
            job = time.time() * 10000000
            if job in prev:
                time.sleep(1)
                job = time.time() * 10000000
            prev[job] = True
            e['SLURM_JOBID'] = f"{job}"
            x = subprocess.run(args=[tasks[exp], ">", f"/tiger/u/lando/jobs/slurm-{job}.out", "2>", "/tiger/u/lando/jobs/slurm-{job}.out" "&"], env=e)
            jobs.append(job)
    print(jobs)


if __name__ == '__main__':
    for task in vels:
        if task not in tasks:
            raise Exception(f"internal error, tasks inconsistent: {task} missing")
    main(sys.argv[1])
