# -*- coding: utf-8 -*-

# Kivy things
import kivy
kivy.require('1.10.1')

from kivy.app import App

from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.treeview import TreeView, TreeViewLabel
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from kivy.lang import Builder
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.label import Label
from kivy.properties import BooleanProperty
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty

from kivy.config import Config

# Other imports
from os import scandir, getcwd

Config.set('graphics', 'width', 1280)
Config.set('graphics', 'height', 720)

class frame_list_tree(FloatLayout):
    def __init__(self, route,**kwargs):
        super(frame_list_tree, self).__init__(**kwargs)

        tv = TreeView(root_options=dict(text='Tree One'),
                      hide_root=True,
                      indent_level=4)

        tree = self.parse_list_to_tree(self.sort_frames_list(self.ls(route)))

        self.populate_tree_view(tv, None, tree)

        self.add_widget(tv)

    def populate_tree_view(self, tree_view, parent, node):
        if parent is None:
            tree_node = tree_view.add_node(TreeViewLabel(text=node['text'],
                                                        is_open=True))
        else:
            tree_node = tree_view.add_node(TreeViewLabel(text=node['text'],
                                                        is_open=True), parent)

        for child_node in node['children']:
            self.populate_tree_view(tree_view, tree_node, child_node)

    def ls(self, ruta = getcwd()):
        return [arch.name for arch in scandir(ruta) if arch.is_file()]

    def sort_frames_list(self, frames_list):
        crop_name = lambda x: (int(x[6:].split('.')[0]), x)
        list_def = []
        
        for element in frames_list:
            list_def.append(crop_name(element))

        list_def.sort(key=lambda tup: tup[0])

        return list_def

    def parse_list_to_tree(self, list_to_parse):
        tree_aux = []
        for i in range(len(list_to_parse)):
            tree_aux.append({'node_id': list_to_parse[i][0], 'text':list_to_parse[i][1], 'children':[]})
        
        tree = {'node_id': 1, 'text': 'Recorded Frames', 'children': tree_aux}

        return tree

Builder.load_string('''
<SelectableLabel>:
    # Draw a background to indicate selection
    canvas.before:
        Color:
            rgba: (.0, 0.9, .1, .3) if self.selected else (0, 0, 0, 1)
        Rectangle:
            pos: self.pos
            size: self.size
<frames_recycleview>:
    viewclass: 'SelectableLabel'
    SelectableRecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
        multiselect: False
        touch_multiselect: False

<icon_button_next>:
    canvas:
        Rectangle:
            source:self.icon
            pos: self.center
            size: 30,30

<icon_button_prev>:
    canvas:
        Rectangle:
            source:self.icon
            pos: self.center
            size: 30,30
<icon_button_mark>:
    canvas:
        Rectangle:
            source:self.icon
            pos: self.center
            size: 30,30

<icon_button_dir>:
    canvas:
        Rectangle:
            source:self.icon
            pos: self.center
            size: 30,30
''')
class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    ''' Adds selection and focus behaviour to the view. '''

class SelectableLabel(RecycleDataViewBehavior, Label):
    ''' Add selection support to the Label '''
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index  = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        ''' Respond to the selection of items in the view. '''
        self.selected = is_selected
        if is_selected:
            print("selection changed to {} ()".format(rv.data[index]['name']))
        else:
            print("selection removed for {} ()".format(rv.data[index]['name']))

class frames_recycleview(RecycleView):
        def __init__(self, route, **kwargs):
            super(frames_recycleview, self).__init__(**kwargs)
            frames_list = self.sort_frames_list(self.ls(route))
            self.data = [{'text': str(x[0]), 'name': x[1]} for x in frames_list]

        def ls(self, ruta = getcwd()):
            return [arch.name for arch in scandir(ruta) if arch.is_file()]

        def sort_frames_list(self, frames_list):
            crop_name = lambda x: (int(x[6:].split('.')[0]), x)
            list_def = []
            
            for element in frames_list:
                list_def.append(crop_name(element))

            list_def.sort(key=lambda tup: tup[0])

            return list_def

class icon_button_next(Button):
    icon = "icons/next.png"

class icon_button_prev(Button):
    icon = "icons/previous.png"

class icon_button_mark(Button):
    icon = "icons/mark.png"

class icon_button_dir(Button):
    icon = "icons/dir.png"
    
class button_container(GridLayout):
    def __init__(self, icon_route, **kwargs):
        super(button_container, self).__init__(**kwargs)
        self.cols = 4
        
        self.button_next = icon_button_next()
        self.button_prev = icon_button_prev()
        self.button_mark = icon_button_mark()
        self.button_dir  = icon_button_dir()

        self.add_widget(self.button_prev)
        self.add_widget(self.button_next)
        # self.add_widget(self.button_mark)
        # self.add_widget(self.button_dir)

class image_container(GridLayout):
    def __init__(self, route, **kwargs):
        super().__init__(**kwargs)
        self.cols = 1

        self.actual_img = Image(source = "{}/frame_0.jpg".format(route))
        self.add_widget(self.actual_img)

class MainWindow(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2

        self.frames_list_rv = frames_recycleview(route="/home/alfonso/Documentos/Proyectos/movida/Clasificacion-de-las-fases-de-la-marcha/src/raspberry_stuff/recorded_frames")
        self.add_widget(self.frames_list_rv)

        self.images = image_container(route="/home/alfonso/Documentos/Proyectos/movida/Clasificacion-de-las-fases-de-la-marcha/src/raspberry_stuff/recorded_frames")
        self.add_widget(self.images)

        self.button_bar = button_container(icon_route="icons")
        self.add_widget(self.button_bar)
        # hacer que los botones:
        # -> pasen imagenes
        # -> marquen
        # -> permitan elegir la ruta

        # Retocar los tamaños de las cosas para que tenga mejor pinta

        self.layout_images = GridLayout()
        self.layout_images.cols = 2


class EpicApp(App):
    title = "KNIGHT"
    
    def build(self):
        return MainWindow()
    
if __name__ == '__main__':
    EpicApp().run()