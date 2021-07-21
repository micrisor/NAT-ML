from matplotlib import rcParams

def applyPlotStyle():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 18
    rcParams['axes.linewidth'] = 2
    rcParams['grid.linewidth'] = 2
    rcParams['grid.color'] = 'gainsboro'
    rcParams['font.weight'] = 'normal'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.labelsize'] = 21
    rcParams['axes.titlesize'] = 18
    rcParams['axes.titleweight'] = 'bold'
    rcParams['legend.edgecolor'] = 'none'
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False


