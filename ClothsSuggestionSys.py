
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

weather = ctrl.Antecedent(np.arange(-30, 60, 0.1), 'weather')
bloodSugar = ctrl.Antecedent(np.arange(30, 1000,0.1), 'bloodSugar')
wool = ctrl.Consequent(np.arange(0, 100, 0.1), 'wool')
cotton = ctrl.Consequent(np.arange(0, 100,0.1), 'cotton')
color = ctrl.Consequent(np.arange(0,255, 0.1), 'color')

weather['cold'] = fuzz.zmf(weather.universe, -30,15)
weather['warm'] = fuzz.gaussmf(weather.universe, 15,25)
weather['hot'] = fuzz.smf(weather.universe, 25, 60)

bloodSugar['hypoglycemia'] = fuzz.zmf(bloodSugar.universe, 0, 70)
bloodSugar['normal'] = fuzz.gaussmf(bloodSugar.universe, 70, 100)
bloodSugar['hyperglycemia'] = fuzz.smf(bloodSugar.universe, 100, 1000)

wool['low'] = fuzz.trimf(wool.universe, [0, 0, 33])
wool['medium'] = fuzz.trimf(wool.universe, [0, 33, 66])
wool['high'] = fuzz.trimf(wool.universe, [33, 66, 100])

cotton['low'] = fuzz.trimf(cotton.universe, [0, 0, 33])
cotton['medium'] = fuzz.trimf(cotton.universe, [0, 33, 66])
cotton['high'] = fuzz.trimf(cotton.universe, [33, 66, 100])

color['black'] = fuzz.trimf(color.universe, [0, 0, 100])
color['greys'] = fuzz.trimf(color.universe, [100,100,200])
color['white'] = fuzz.trimf(color.universe, [150,150,255])

weather.view()
plt.title('weather')
plt.show()

bloodSugar.view()
plt.title('bloodSugar')
plt.show()

wool.view()
plt.title('wool')
plt.show()

cotton.view()
plt.title('cotton')
plt.show()

color.view()
plt.title('color')
plt.show()

rule1 = ctrl.Rule(weather['cold'] & bloodSugar['normal'],wool['high'])
rule2 = ctrl.Rule(weather['cold'] & bloodSugar['hypoglycemia'],wool['high'])
rule3 = ctrl.Rule(weather['cold'] & bloodSugar['hyperglycemia'],wool['high'])
rule4 = ctrl.Rule(weather['warm'] & bloodSugar['normal'], wool['medium'])
rule5 = ctrl.Rule(weather['warm'] & bloodSugar['hypoglycemia'], wool['high'])
rule6 = ctrl.Rule(weather['warm'] & bloodSugar['hyperglycemia'], wool['medium'])
rule7 = ctrl.Rule(weather['hot'] & bloodSugar['normal'], wool['low'])
rule8 = ctrl.Rule(weather['hot'] & bloodSugar['hypoglycemia'], wool['low'])
rule9 = ctrl.Rule(weather['hot'] & bloodSugar['hyperglycemia'], wool['low'])

rule10 = ctrl.Rule(weather['cold'] & bloodSugar['normal'], cotton['low'])
rule11 = ctrl.Rule(weather['cold'] & bloodSugar['hypoglycemia'], cotton['low'])
rule12 = ctrl.Rule(weather['cold'] & bloodSugar['hyperglycemia'], cotton['low'])
rule13 = ctrl.Rule(weather['warm'] & bloodSugar['normal'], cotton['medium'])
rule14 = ctrl.Rule(weather['warm'] & bloodSugar['hypoglycemia'], cotton['low'])
rule15 = ctrl.Rule(weather['warm'] & bloodSugar['hyperglycemia'], cotton['medium'])
rule16 = ctrl.Rule(weather['hot'] & bloodSugar['normal'], cotton['high'])
rule17 = ctrl.Rule(weather['hot'] & bloodSugar['hyperglycemia'], cotton['high'])
rule18 = ctrl.Rule(weather['hot'] & bloodSugar['hypoglycemia'], cotton['high'])

rule19 = ctrl.Rule(weather['cold'] & bloodSugar['normal'], color['black'])
rule20 = ctrl.Rule(weather['cold'] & bloodSugar['hypoglycemia'], color['black'])
rule21 = ctrl.Rule(weather['cold'] & bloodSugar['hyperglycemia'], color['black'])
rule22 = ctrl.Rule(weather['hot'] &bloodSugar['normal'], color['white'])
rule23 = ctrl.Rule(weather['hot'] &bloodSugar['hypoglycemia'], color['white'])
rule24 = ctrl.Rule(weather['hot'] &bloodSugar['hyperglycemia'], color['white'])
rule25 = ctrl.Rule(weather['warm'] & bloodSugar['normal'], color['greys'])
rule26= ctrl.Rule(weather['warm'] & bloodSugar['hyperglycemia'], color['greys'])
rule27 = ctrl.Rule(weather['warm'] & bloodSugar['hypoglycemia'], color['black'])

suggestion_ctrl = ctrl.ControlSystem([ rule1, rule2,rule3,rule4, rule5 ,rule6,rule7,rule8,rule9,
                                       rule10, rule11, rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule18,
                                       rule19,rule20,rule21,rule22,rule23,rule24,rule25,rule26,rule27])

suggestion = ctrl.ControlSystemSimulation(suggestion_ctrl)

suggestion.input['bloodSugar'] = 70
suggestion.input['weather'] = -20
suggestion.compute()

print(suggestion.output['wool'])
print(suggestion.output['cotton'])
print(suggestion.output['color'])
wool.view(sim=suggestion)
plt.title('Result for wool')
plt.show()
cotton.view(sim=suggestion)
plt.title('Result for cotton')
plt.show()
color.view(sim=suggestion)
plt.title('Result for color')
plt.show()






