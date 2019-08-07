import re
import cv2
import time
from copy import deepcopy
from src.utils.utils import *


class Station:
    def __init__(self):
        self.station_index = 0
        self.station_box = 0
        self.station_node = 0
        self.processing_count = 1000
        self.is_free = 1
        self.counter = 0
        self.car_id = -1


class Car:
    def __init__(self, sv):
        self.sv = sv
        self.car_id = 0
        self.source_node = 0
        self.color = 0
        self.move_counter_maxvalue = 0
        self.destination_index = 0
        self.destination_choose_array = []
        self.temp_source_node = 0
        self.training_done = 0
        self.training_done_count = 5
        self.destination_node = None

    def increase_destination_index(self):
        self.destination_index += 1
        self.destination_index %= len(self.sv.destination_nodes_list)


class Configuration:
    """
    This class is used to create environment
    It also creates methods like, get_legal_actions, get_successor_node etc.
    """

    def __init__(self, env_config):
        environment_config = env_config
        self.disconnected_nodes_list = environment_config['disconnected_nodes_list']
        self.environment_rows = environment_config["number_of_rows"]
        self.environment_columns = environment_config["number_of_cols"]
        self.environment_bad_states_file = environment_config["bad_states_file"]
        self.initial_states = environment_config["initial_states"]
        self.no_box_list = environment_config["no_box_list"]  # write number of boxes which you want to set as no_boxes
        self.no_node_list = environment_config["no_node_list"]  # write number of nodes that you want to set as no_nodes
        self.no_floor = environment_config["no_floor"]
        self.box_node_list = {}
        self.environment_node_name_list = environment_config["environment_node_name_list"]
        self.action_dictionary = {}
        self.blocked_nodes_list = []
        self.training_complete_destination_index = environment_config["training_complete_destination_index"]
        self.selective_blocked_nodes_dictionary = environment_config["selective_blocked_nodes_dictionary"]
        self.selective_blocked_boxes_dictionary = environment_config["selective_blocked_boxes_dictionary"]
        """
        No_node_boxes are those boxes where there does not exist any physical path.
        """
        self.is_destination_node = environment_config["is_destination_node"]
        """
        if destination is defined in node then
        flag is_destination_node will be true
        if destination is defined in box then
        flag will be false
        """

        ####################################################################
        """
        ALl the information about station is defined below.
        """
        ####################################################################
        self.no_of_stations = environment_config["no_of_stations"]
        self.station_box_list = environment_config["station_box_list"]
        self.destination_nodes_list = environment_config["station_destination_list"]
        self.station_processing_count_list = environment_config["station_processing_count_list"]
        # define which node of station is destination node. If whole box can be defined as destination box,
        # then write only name of box.
        # You are supposed to write name of destinations in a particular order.
        self.station_entrance_list = environment_config["station_entrance_list"]
        # This list defines that from which side you can enter in the station. If station has multiple entries,
        # describe it in a nested lists.

        ####################################################################
        """
        ALl the information about car is defined below.
        """
        ####################################################################
        self.no_of_cars = environment_config["no_of_cars"]
        self.cars_info = environment_config["cars_info"]
        self.disconnected_nodes_list = environment_config["disconnected_nodes_list"]
        """
        This code is creating box_list
        """
        self.box_list = []
        for i in range(0, self.environment_columns):
            for j in range(0, self.environment_rows):
                box_number = 'U' + str(self.rowcol_to_num(j, i))
                self.box_list.append(box_number)
            for i in range(0, self.environment_columns):
                for j in range(0, self.environment_rows):
                    box_number = 'D' + str(self.rowcol_to_num(j, i))
                    self.box_list.append(box_number)
        print(self.box_list)

        """
        This code is creating node_list related to the nodes
        """
        for i in self.box_list:
            node_list = []
            for j in range(0, 4):
                node_string = str(i) + self.environment_node_name_list[j]
                node_list.append(node_string)
                self.box_node_list[i] = node_list

        """
        This code is creating_node_list and it's relevant action list
        """
        for i in self.box_node_list:
            for j in self.box_node_list[i]:
                # nodes are arranged according to actions Stop Left Right Front Back TurnBack
                action_specific_node_list = [j, self.give_left_action_node(j), self.give_right_action_node(j),
                                             self.give_front_action_node(j), self.give_back_action_node(j),
                                             self.give_turnback_action_node(j)]
                self.action_dictionary[j] = action_specific_node_list
                # if j == 'U8E':
                #     print(action_specific_node_list)
                #     input()
        # print (self.action_dictionary)

        '''
        This code is for creating no_node_list
        '''
        for i in self.no_box_list:
            nodes = self.box_node_list[i]
            for j in nodes:
                self.no_node_list.append(j)

        self.stations_info = []
        for i in range(0, self.no_of_stations):
            self.stations_info.append([self.station_box_list[i], 0, self.station_processing_count_list[i]])

    def give_left_action_node(self, node):
        index = self.environment_node_name_list.index(node[-1])
        index -= 1
        return node.replace(node[-1], self.environment_node_name_list[index])

    def give_right_action_node(self, node):
        index = self.environment_node_name_list.index(node[-1])
        index += 1
        index %= len(self.environment_node_name_list)
        return node.replace(node[-1], self.environment_node_name_list[index])

    def give_front_action_node(self, node):
        number = int(re.search(r'\d+', node).group())
        # print number, node[-2], int(re.findall(r'\d+', node)[0]) # some other methods
        # if node is in the station
        for num, i in enumerate(self.station_box_list):
            if (number == int(re.search(r'\d+', i).group())) and (i[0] == node[0]):
                print("node: %s is in station" % node)
                entry_array = self.station_entrance_list[num]
                if node[-1] in entry_array:
                    break
                else:
                    return node[0] + 'YY'
        # completes
        row, col = self.num_to_rowcol(number)
        if node[-1] == 'N':
            row -= 1
        if node[-1] == 'S':
            row += 1
        if node[-1] == 'E':
            col += 1
        if node[-1] == 'W':
            col -= 1
        # if node exists
        if not (row == self.environment_rows or row == -1 or col == self.environment_columns or col == -1):
            box_num = self.rowcol_to_num(row, col)
            output_node = node.replace(str(number), str(box_num))
            # Check if node is disconnected
            # if output_node in self.disconnected_nodes_list: return node[0] + 'YY'
            if output_node in self.disconnected_nodes_list:
                return node[0] + 'YY'
            output_box = int(re.search(r'\d+', output_node).group())
            # if node is in the station
            for num, i in enumerate(self.station_box_list):
                if (output_box == int(re.search(r'\d+', i).group())) and (i[0] == output_node[0]):
                    entry_array = self.station_entrance_list[num]
                    opposite_entry_array = []
                    for i in entry_array:
                        opposite_entry_array.append(self.opposite_action(i))
                    if output_node[-1] in opposite_entry_array:
                        break
                    else:
                        return node[0] + 'YY'
            return output_node
        else:
            if node[0] == 'U':
                return 'UYY'
            else:
                return 'DYY'

    def give_back_action_node(self, node):
        number = int(re.search(r'\d+', node).group())
        # if node is in the station
        for num, i in enumerate(self.station_box_list):
            if (number == int(re.search(r'\d+', i).group())) and (i[0] == node[0]):
                print("node is in station")
                entry_array = self.station_entrance_list[num]
                opposite_entry_array = []
                for i in entry_array:
                    opposite_entry_array.append(self.opposite_action(i))
                if node[-1] in opposite_entry_array:
                    break
                else:
                    return node[0] + 'YY'
        # completes
        row, col = self.num_to_rowcol(number)
        if node[-1] == 'N':
            row += 1
        if node[-1] == 'S':
            row -= 1
        if node[-1] == 'E':
            col -= 1
        if node[-1] == 'W':
            col += 1
        # if node exists
        if not (row == self.environment_rows or row == -1 or col == self.environment_columns or col == -1):
            box_num = self.rowcol_to_num(row, col)
            output_node = node.replace(str(number), str(box_num))
            output_box = int(re.search(r'\d+', output_node).group())
            # if node is in the station
            for num, i in enumerate(self.station_box_list):
                if (output_box == int(re.search(r'\d+', i).group())) and (i[0] == output_node[0]):
                    entry_array = self.station_entrance_list[num]
                    if output_node[-1] in entry_array:
                        break
                    else:
                        return node[0] + 'YY'
            return output_node
        else:
            if node[0] == 'U':
                return 'UYY'
            else:
                return 'DYY'

    def give_turnback_action_node(self, node):
        index = self.environment_node_name_list.index(node[-1])
        index -= 2
        return node.replace(node[-1], self.environment_node_name_list[index])

    def num_to_rowcol(self, number):
        col = (number - 1) // self.environment_rows
        row = (number - 1) % self.environment_rows
        return row, col

    def rowcol_to_num(self, row, col):
        return col * self.environment_rows + (row + 1)

    def opposite_action(self, action):
        opp_actions = {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}
        return opp_actions[action]


class Display:
    def __init__(self, env_conf):
        self.env_conf = env_conf
        self.rows = self.env_conf.environment_rows
        self.columns = self.env_conf.environment_columns
        self.disconnected_nodes_list = self.env_conf.disconnected_nodes_list
        self.box_length = -1
        self.box_width = -1
        self.radius = -1
        self.margin = 100
        self.static_display_down = np.zeros((1080, 1920, 3), np.uint8)
        self.static_display_up = np.zeros((1080, 1920, 3), np.uint8)
        self.static_display_up[:, :] = [255, 255, 255]
        self.static_display_down[:, :] = [255, 255, 255]
        self.dynamic_display_down = np.zeros((700, 1200, 3), np.uint8)
        self.dynamic_display_up = np.zeros((700, 1200, 3), np.uint8)
        self.line_colour = (0, 0, 255)
        self.circle_colour = (0, 0, 255)
        self.line_thickness = 1
        self.circle_thickness = 1
        self.text_size = 0.5
        self.no_box_color = (0, 0, 0)
        self.no_box_thickness = -1
        self.no_box_size_fraction = 1
        self.station_box_size_fraction = 1
        self.station_box_thickness = 10
        self.station_box_color = (0, 127, 0)
        self.lift_table_box_size_fraction = 0.5
        self.lift_table_box_thickness = -1
        self.lift_table_box_color = (0, 255, 0)

        self.scalable_static_background()

    def scalable_static_background(self):

        for box in self.env_conf.station_box_list:
            self.static_display_up, self.static_display_down = self.color_for_stations(self.static_display_up,
                                                                                       self.static_display_down, box,
                                                                                       self.station_box_color, 1)

        # mode is a check value that tells us about the up/down configuration
        mode = self.env_conf.station_box_list[0][0]
        self.lines_and_circles_for_display(mode)
        for box in self.env_conf.no_box_list:
            self.static_display_up, self.static_display_down = self.color_for_no_box(self.static_display_up,
                                                                                     self.static_display_down, box,
                                                                                     self.no_box_color)

    def lines_and_circles_for_display(self, mode):
        # The below function is a single implementation for the up/down configurations displaying lines and circles
        if mode == 'U':
            img = self.static_display_up
        else:
            img = self.static_display_down
        rows = self.rows
        columns = self.columns
        length = img.shape[0]
        width = img.shape[1]
        margin = self.margin
        length -= 2 * margin
        width -= 2 * margin
        starting_x = margin
        starting_y = margin
        optimum_box_length = length / 3
        optimum_box_width = width / 5
        box_length = int((3 * optimum_box_length) / rows)
        box_width = int((5 * optimum_box_width) / columns)
        self.box_length = box_length
        self.box_width = box_width
        row_ratio = float(3.0 / rows)
        column_ratio = float(5.0 / columns)

        for row in range(rows + 1):
            cv2.line(img, (starting_x, starting_y + row * box_length),
                     (starting_x + width, starting_y + row * box_length), (0, 255, 0), 3)
        for col in range(columns + 1):
            cv2.line(img, (starting_x + col * box_width, starting_y),
                     (starting_x + col * box_width, starting_y + length), (0, 255, 0), 3)

        optimum_radius = 40
        radius = 0
        for row in range(rows):
            for col in range(columns):
                mid_x = starting_x + int(col * box_width + box_width / 2)
                mid_y = starting_y + int(row * box_length + box_length / 2)
                length_radius = int(optimum_radius * row_ratio)
                width_radius = int(optimum_radius * column_ratio)
                radius = min(length_radius, width_radius)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.circle(img, (starting_x + col * box_width + 2 * radius, mid_y), radius, self.line_colour, 3)
                cv2.putText(img, 'W', (starting_x + col * box_width + 2 * radius - radius // 2, mid_y + radius // 2),
                            font, self.text_size, (0, 0, 0))
                cv2.circle(img, (starting_x + (col + 1) * box_width - 2 * radius, mid_y), radius, self.line_colour, 3)
                cv2.putText(img, 'E',
                            (starting_x + (col + 1) * box_width - 2 * radius - radius // 2, mid_y + radius // 2), font,
                            self.text_size, (0, 0, 0))
                cv2.circle(img, (mid_x, starting_y + row * box_length + 2 * radius), radius, self.line_colour, 3)
                cv2.putText(img, 'N', (mid_x - radius // 2, starting_y + row * box_length + 2 * radius + radius // 2),
                            font, self.text_size, (0, 0, 0))
                cv2.circle(img, (mid_x, starting_y + (row + 1) * box_length - 2 * radius), radius, self.line_colour, 3)
                cv2.putText(img, 'S',
                            (mid_x - radius // 2, starting_y + (row + 1) * box_length - 2 * radius + radius // 2), font,
                            self.text_size, (0, 0, 0))
        self.radius = radius

    def node_to_coordinate(self, node):

        number_in_string = re.search(r'\d+', node).group()
        number = int(number_in_string)
        node_without_number = "".join(node.split(number_in_string))
        node_direction = node_without_number[1]
        col = (number - 1) // self.rows
        row = (number - 1) % self.rows
        starting_x = self.margin
        starting_y = self.margin
        mid_x = int(starting_x + col * self.box_width + self.box_width / 2)
        mid_y = int(starting_y + row * self.box_length + self.box_length / 2)
        offset_x = int(self.box_width / 2 - 2 * self.radius)
        offset_y = int(self.box_length / 2 - 2 * self.radius)
        if node_direction == 'E':
            mid_x += offset_x
        if node_direction == 'N':
            mid_y -= offset_y
        if node_direction == 'W':
            mid_x -= offset_x
        if node_direction == 'S':
            mid_y += offset_y
        return [mid_x, mid_y]

    def draw_triangle_on_node(self, img_up, img_down, node, triangle_color):
        c1 = self.node_to_coordinate(node[0])
        c2 = self.node_to_coordinate(node[1])
        c3 = self.node_to_coordinate(node[2])
        pts = np.array([c1, c2, c3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        if node[0][0] == 'D':
            img_down = cv2.polylines(img_down, [pts], True, triangle_color, 5)
        if node[0][0] == 'U':
            img_up = cv2.polylines(img_up, [pts], True, triangle_color, 5)
        return img_up, img_down

    def color_for_no_box(self, img_up, img_down, boxno, box_color):
        number = int(re.search(r'\d+', boxno).group())
        col = (number - 1) // self.rows
        row = (number - 1) % self.rows
        vertex1 = (self.margin + col * self.box_width, self.margin + row * self.box_length)
        vertex2 = (self.margin + (col + 1) * self.box_width, self.margin + (row + 1) * self.box_length)
        if boxno[0] == 'U':
            img_up = cv2.rectangle(img_up, vertex1, vertex2, box_color, -1)
        else:
            img_down = cv2.rectangle(img_down, vertex1, vertex2, box_color, -1)
        return img_up, img_down

    def color_for_stations(self, img_up, img_down, boxno, box_color, thickness):
        number = int(re.search(r'\d+', boxno).group())
        col = (number - 1) // self.rows
        row = (number - 1) % self.rows
        vertex1 = (self.margin + col * self.box_width, self.margin + row * self.box_length)
        vertex2 = (self.margin + (col + 1) * self.box_width, self.margin + (row + 1) * self.box_length)
        if boxno[0] == 'U':
            img_up = cv2.rectangle(img_up, vertex1, vertex2, box_color, thickness)
        else:
            img_down = cv2.rectangle(img_down, vertex1, vertex2, box_color, thickness)
        return img_up, img_down

    def update(self, cars, stations, waitkey=25):
        self.dynamic_display_up = deepcopy(self.static_display_up)
        self.dynamic_display_down = deepcopy(self.static_display_down)
        time.sleep(0.5)

        for num, i in enumerate(cars):
            self.dynamic_display_up, self.dynamic_display_down = self.draw_triangle_on_node(self.dynamic_display_up,
                                                                                            self.dynamic_display_down, i,
                                                                                            self.env_conf.cars_info[num][
                                                                                               1])
        # stations which are occupied make it blue
        for num, i in enumerate(stations):
            self.dynamic_display_up, self.dynamic_display_down = self.color_for_stations(self.dynamic_display_up,
                                                                                         self.dynamic_display_down,
                                                                                         self.env_conf.stations_info[
                                                                                             num][0],
                                                                                         (255, 128, 128),
                                                                                         self.station_box_thickness)
        if waitkey == 3000:
            font = cv2.FONT_HERSHEY_SIMPLEX
            buf = "Deadlock"
            print(buf)
            cv2.putText(self.dynamic_display_down, buf, (500, 300), font, 5, (255, 0, 0),
                        5, cv2.LINE_AA)
            buf = "Detected"
            print(buf)
            cv2.putText(self.dynamic_display_down, buf, (500, 600), font, 5, (255, 0, 0),
                        5, cv2.LINE_AA)
        cv2.namedWindow('down', cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow('up', cv2.WINDOW_GUI_EXPANDED)

        cv2.imshow('up', self.dynamic_display_up)
        cv2.imshow('down', self.dynamic_display_down)
        return cv2.waitKey(waitkey)

