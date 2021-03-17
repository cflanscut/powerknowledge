"""
此文档用于修改json文件但不修改其格式，主要在meta字典中添加更多label信息
运行此文档更新json文件
要新添加一些简单规则的label可在此文档修改，但复杂的仍需手工打标签
"""
import json
import copy


def add_is_heat(app):
    """
    为输入的app信息添加该电器是否加热的信息标签
    :param app: 输入的某一电器数据信息，为dict格式
    :return: none
    """
    heat_type = [
        "Coffee maker", "Hair Iron", "Heater", "Incandescent Light Bulb",
        "Microwave", "Soldering Iron", "Water kettle"
    ]
    if app["appliance"]["type"] in heat_type:
        app["extra label"]["is_heat"] = "1"
    elif app["appliance"]["type"] == 'Hairdryer':
        if app["appliance"]["status"] == "highhot" or app["appliance"][
                "status"] == "lowhot":
            app["extra label"]["is_heat"] = "1"
        else:
            app["extra label"]["is_heat"] = "0"
    else:
        app["extra label"]["is_heat"] = "0"


def add_is_cool(app):
    """
    为输入的app信息添加该电器是否制冷的信息标签
    :param app: 输入的某一电器数据信息，为dict格式
    :return: none
    """
    cool_type = ["Fridge"]
    if app["appliance"]["type"] in cool_type:
        app["extra label"]["is_cool"] = "1"
    elif app["appliance"]["type"] == 'Air Conditioner':
        if app["appliance"]["status"] == "highfan" or app["appliance"][
                "status"] == "lowfan":
            app["extra label"]["is_cool"] = "0"
        else:
            app["extra label"]["is_cool"] = "1"
    else:
        app["extra label"]["is_cool"] = "0"


def add_is_rotate(app):
    """
    为输入的app信息添加该电器是否旋转的信息标签
    :param app: 输入的某一电器数据信息，为dict格式
    :return: none
    """
    rotate_type = [
        "Air Conditioner", "Blender", "Fan", "Vacuum", "Hairdryer",
        "Washing Machine"
    ]
    if app["appliance"]["type"] in rotate_type:
        app["extra label"]["is_rotate"] = "1"
    else:
        app["extra label"]["is_rotate"] = "0"


def add_is_light(app):
    """
    为输入的app信息添加该电器是否发光的信息标签
    :param app: 输入的某一电器数据信息，为dict格式
    :return: none
    """
    light_type = [
        'Compact Fluorescent Lamp', "Incandescent Light Bulb", "Laptop"
    ]
    if app["appliance"]["type"] in light_type:
        app["extra label"]["is_light"] = "1"
    else:
        app["extra label"]["is_light"] = "0"


def add_module(app):
    machinery_module = [
        'Air Conditioner', 'Vacuum', 'Fan', 'Washing Machine', 'Blender'
    ]
    heater_module = [
        'Microwave', "Coffee maker", "Hair Iron", "Heater",
        "Incandescent Light Bulb", "Soldering Iron", "Water kettle"
    ]
    electronical_module = ['Compact Fluorescent Lamp', 'Laptop']
    hairdryer_heater_mode = [
        'highhot', 'lowhot', 'highheat', 'lowheat', 'mediumhot', 'highwarm',
        'lowwarm'
    ]
    hairdryer_machin_mode = ['off-on', 'highcool', 'lowcool', 'noheat']

    tp = app["appliance"]["type"]
    if tp in machinery_module:
        app["extra label"]["module"] = "machinery"
    if tp in heater_module:
        app["extra label"]["module"] = "heater"
    if tp in electronical_module:
        app["extra label"]["module"] = "electronical"
    if tp == 'Fridge':
        if app["appliance"]["status"] == 'off-on':
            app["extra label"]["module"] = "machinery"
        if app["appliance"]["status"] == 'off-defroster':
            app["extra label"]["module"] = "heater"
    if tp == 'Hairdryer':
        if app["appliance"]["status"] in hairdryer_heater_mode:
            app["extra label"]["module"] = "heater"
        if app["appliance"]["status"] in hairdryer_machin_mode:
            app["extra label"]["module"] = "machinery"


meta_path = 'data/source/'
meta_name = 'metadata_submetered.json'
outp_name = 'metadata_submetered2.1.json'
with open(meta_path + meta_name) as data_file:
    meta = json.load(data_file)
meta_temp = copy.deepcopy(meta)

###################################
for i in range(1, len(meta_temp) + 1):
    i_s = str(i)
    meta_temp[i_s]["extra label"] = {}
    # 功能（自己定）
    add_is_heat(meta_temp[i_s])
    add_is_cool(meta_temp[i_s])
    add_is_rotate(meta_temp[i_s])
    add_is_light(meta_temp[i_s])
    add_module(meta_temp[i_s])

###############################
print("XXX")
with open(meta_path + outp_name, 'w') as json_file:
    json.dump(meta_temp, json_file, indent=4)
