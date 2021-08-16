from pathlib import Path
import math

perspective = 30 / 180 * math.pi
conv_layers = [
    {
        'height': 144.0,
        'width': 256.0,
        'channels': 9.0
    },
    {
        'height': 71.0,
        'width': 127.0,
        'channels': 32.0
    },
    {
        'height': 34.0,
        'width': 62.0,
        'channels': 64.0
    },
    {
        'height': 16.0,
        'width': 30.0,
        'channels': 128.0
    },
    {
        'height': 7.0,
        'width': 14.0,
        'channels': 256.0
    },
]
depth_shift = 0
layer_shift = [0, 0]
max_x = 0
max_y = 0
for i_layer, layer in enumerate(conv_layers):
    layer_shift[0] = depth_shift + (conv_layers[0]['width'] / 2 - layer['width'] / 2) * math.cos(perspective)
    layer_shift[1] = conv_layers[0]['height'] / 2 - layer['height'] / 2 + (conv_layers[0]['width'] / 2 - layer['width'] / 2) * math.sin(perspective)
    depth_shift += layer['channels']
    if layer_shift[0] + layer['channels'] + layer['width'] * math.cos(perspective) > max_x:
        max_x = layer_shift[0] + layer['channels'] + layer['width'] * math.cos(perspective)
    
    if layer_shift[1] + layer['height'] + layer['width'] * math.sin(perspective) > max_y:
        max_y = layer_shift[1] + layer['height'] + layer['width'] * math.sin(perspective)

depth_shift = 0
layer_shift = [0, 0]

shape = '<shape aspect="variable" h="' + str(max_y) + '" w="' + str(max_x) + '" strokewidth="inherit">\n'
shape += '  <background>\n'
shape += '  </background>\n'
shape += '  <foreground>\n'
for i_layer, layer in enumerate(conv_layers):
    layer_shift[0] = depth_shift + (conv_layers[0]['width'] / 2 - layer['width'] / 2) * math.cos(perspective)
    layer_shift[1] = conv_layers[0]['height'] / 2 - layer['height'] / 2 + (conv_layers[0]['width'] / 2 - layer['width'] / 2) * math.sin(perspective)
    shape += '    <fillstroke />\n'
    shape += '    <fillcolor color="#FFFFFF"/>\n'
    shape += '    <path>\n'
    shape += '      <move x="' + str(layer_shift[0]) +'" y="' + str(layer_shift[1] + layer['height'] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['channels']) + '" y="' + str(layer_shift[1] + layer['height'] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['channels']) + '" y="' + str(layer_shift[1] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <line x="' + str(layer_shift[0]) +'" y="' + str(layer_shift[1] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <close/>\n'
    shape += '    </path>\n'
    shape += '    <fillstroke />\n'
    shape += '    <path>\n'
    shape += '      <move x="' + str(layer_shift[0]) +'" y="' + str(layer_shift[1] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['channels']) + '" y="' + str(layer_shift[1] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['channels'] + layer['width'] * math.cos(perspective)) + '" y="' + str(layer_shift[1]) +'"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['width'] * math.cos(perspective)) + '" y="' + str(layer_shift[1]) +'"/>\n'
    shape += '      <close/>\n'
    shape += '    </path>\n'
    shape += '    <fillstroke />\n'
    shape += '    <fillcolor color="#DAE8FC"/>\n'
    shape += '    <path>\n'
    shape += '      <move x="' + str(layer_shift[0] + layer['channels']) + '" y="' + str(layer_shift[1] + layer['height'] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['channels'] + layer['width'] * math.cos(perspective)) + '" y="' + str(layer_shift[1] + layer['height']) + '"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['channels'] + layer['width'] * math.cos(perspective)) + '" y="' + str(layer_shift[1]) +'"/>\n'
    shape += '      <line x="' + str(layer_shift[0] + layer['channels']) + '" y="' + str(layer_shift[1] + layer['width'] * math.sin(perspective)) + '"/>\n'
    shape += '      <close/>\n'
    shape += '    </path>\n'
    depth_shift += layer['channels']

shape += '    <fillstroke />\n'
shape += '  </foreground>\n'
shape += '</shape>\n'
layer_file = open('paper_plots/conv_layers.txt', 'w')
layer_file.write(shape)
layer_file.close()
