feature_select = get_feature_name()
selected_label = [
    'Air Conditioner', 'Blender', 'Coffee maker', 'Fan', 'Fridge', 'Hair Iron',
    'Hairdryer', 'Heater', 'Incandescent Light Bulb', 'Microwave',
    'Soldering Iron', 'Vacuum', 'Washing Machine', 'Water kettle'
]
x_mh, y_mh, mh_index = read_processed_data(
    'type',
    type_header='appliance',
    selected_label=selected_label,
    direaction=1,
    offset=1,
    each_lenth=1,
    feature_select=feature_select,
    source='submetered_process2.1/training')
