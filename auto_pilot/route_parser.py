#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module used to parse all the route and scenario configuration parameters.
"""
import xml.etree.ElementTree as ET

import carla

def parse_routes_file(route_filename):
    list_route_descriptions = []
    tree = ET.parse(route_filename)
    for route in tree.iter("route"):

        route_id = route.attrib['id']

        new_config = {}
        new_config['town'] = route.attrib['town']
        new_config['name'] = "RouteScenario_{}".format(route_id)

        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
                                                y=float(waypoint.attrib['y']),
                                                z=float(waypoint.attrib['z'])))

        new_config['trajectory'] = waypoint_list

        list_route_descriptions.append(new_config)

    return list_route_descriptions
