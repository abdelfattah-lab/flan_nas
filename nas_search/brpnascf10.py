import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Setup Plot Styling
sns.set_palette("tab10")
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 16  # Increase font size

# flan = [(4, 70.34),(8, 70.34),(12, 70.34),(16, 70.34),(24, 71.36),(28, 72.95),(32, 73.49),(48, 73.49),(64, 73.49),(96, 73.49),(128, 73.49)]
flan = [(4, 66.58), (8, 70.87), (12, 70.87),(16, 72.77),(24, 72.77),(28, 72.77),(32, 73.49),(48, 73.49),(64, 73.49),(96, 73.49),(128, 73.49),(192, 73.49)]
brpnas = [(19.20838183934808,71.01009821492082),(19.906868451688013,71.16135283636571),(20.605355064027947,71.17698897792982),(25.494761350407444,71.70372959928103),(28.28870779976718,71.90710874079831),(30.034924330617002,71.92272666549847),(32.13038416763679,71.94355461329067),(33.87660069848661,71.94352425185075),(37.36903376018627,72.00084057815121),(39.464493597206044,72.50154930090615),(41.210710128055894,72.71537885004744),(48.894062863795114,73.05950755479373),(53.78346915017464,73.06463863814194),(58.32363213038417,73.10107236605833),(62.16530849825378,73.32529767223177),(68.80093131548313,73.47123296940087),(77.18277066356228,73.50238380676939),(79.97671711292202,73.57536056),(82.77066356228173,73.62747293928231),(86.26309662398137,73.64306050254255),(90.80325960419091,73.66906203969889),(98.13736903,73.66893452165117),(105.82072176949941,73.75225845739595),(115.9487776484284,73.79381112),(128.87077997671713,73.83009911708932),(141.0942956926659,73.82988658700981),(170.43073341094293,73.84502480095907),(175.66938300349244,73.87101419353942),(178.46332945285215,73.87096561523553),(195.9254947613504,73.88109419159629),(258.0908032596042,73.89044551509484),(283.5855646100116,73.89000223807186)]
aging_evol = [(40.512223515715945,71.13491409447438),(44.70314318975552,71.1609217),(52.73573923166474,71.31204880731589),(61.46682188591387,71.41621890771684),(69.84866123399303,71.74990327712707),(84.51688009313156,72.16693587143405),(103.37601862630964,72.67778531512572),(129.91850989522698,73.13634021),(116.29802095459837,72.91750103),(94.29569266589056,72.45886719),(149.47613504074505,73.33942788637535),(175.66938300349244,73.51631971),(203.60884749708967,73.68274897681947),(234.69150174621652,73.83869140458961),(272.40977881257277,73.94757360046773),(294.0628637951106,74.00457416779292)]
reinforce = [(35.972060535506415,71.02545503123758),(42.60768335273575,71.12966156536645),(55.18044237485449,71.34330287358017),(75.08731082654248,71.48900742380579),(94.29569266589056,71.64515630936745),(113.8533178114086,71.73348988270074),(140.0465658,71.86343684560254),(167.2875436554133,71.96206901936019),(200.81490104772993,72.06580798731412),(232.94528521536668,72.18521953056009),(269.61583236321303,72.29411994330218),(274.8544819557625,72.34618981278268),(290.22118742724103,72.37721920439148),(298.2537834691502,72.38229563714782)]
random = [(37.71827706635624,71.02020857441762),(47.14784633294528,71.23912062860325),(60.069848661233976,71.41624319686878),(78.23050058207218,71.57241029929439),(98.83585564610013,71.77547975412438),(115.59953434225844,71.83778143),(136.9033760186263,71.95738121303495),(167.63678696158323,72.12376190385312),(200.46565774155997,72.21708082562297),(237.48544819557625,72.36248783373728),(270.31431897555296,72.46623894626718),(296.85681024447024,72.49707402466042)]

# flan = [(4, 70.34),(8, 70.34),(12, 70.34),(16, 70.34),(24, 71.36),(28, 72.95),(32, 73.49),(48, 73.49),(64, 73.49),(96, 73.49),(128, 73.49), (192, 73.49), (256, ), (320, )]
# flan CAZ
# flant
# flant CAZ
# Create a list of markers matplotlib
markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "H", "X", "D", "d", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "X", "D", "d", "1", "2", "3", "4", "8"]
# Create the Plot
plt.figure(figsize=(12, 5))
for idx, lst in enumerate([flan, brpnas, aging_evol, reinforce, random]):
    x_values = [x for x, y in lst]
    y_values = [y for x, y in lst]
    plt.plot(x_values, y_values, marker=markers[idx], linestyle='-', linewidth=4, markersize=15, label=['FLAN', 'BRP-NAS', 'Aging Evolution', 'REINFORCE', 'Random'][idx])

# Set the x-axis to logarithmic scale
# plt.xscale('log', basex=2)
plt.xlim(4, 64)
plt.ylim(70, 74)
plt.gca().get_xaxis().set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

# Add Labels and Legends
# plt.title("Comparison of Different Lists")
plt.xlabel("Trained Models", fontsize=20)
plt.ylabel("Avg. Best Test Accuracy [%]", fontsize=20)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=5)


plt.grid(True, which="both", ls="--", c='0.7')  # Add a grid for better readability
# Adjust layout and save the plot
# plt.subplots_adjust(top=0.75, wspace=0.3)
plt.subplots_adjust(top=0.86, bottom=0.15)
plt.savefig("list_comparison_plot.png")
plt.savefig("list_comparison_plot.pdf")

# Show the plot
plt.show()