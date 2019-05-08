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
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup

from kivy.config import Config

# Other imports
import os
import time

import routes

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

################# KIVY ################### 

Builder.load_string('''
<SelectableLabel>:
    # Draw a background to indicate selection
    canvas.before:
        Color:
            rgba:(.0, 0.9, .1, .3) if self.selected else ( (.9, .0, .1, .3) if self.talon else ((.0, .0, .9, .3)if self.puntera else (0, 0, 0, 1)))

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
<Root>:
    text_input: text_input

    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: 'Load'
                on_release: root.show_load()
            Button:
                text: 'Save'
                on_release: root.show_save()

        BoxLayout:
            TextInput:
                id: text_input
                text: ''

            RstDocument:
                text: text_input.text
                show_errors: True
<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileChooserListView:
            id: filechooser

        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: "Cancel"
                on_release: root.cancel()

            Button:
                text: "Load"
                on_release: root.load(filechooser.path, filechooser.selection)
<SaveDialog>:
    text_input: text_input
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileChooserListView:
            id: filechooser
            on_selection: text_input.text = self.selection and self.selection[0] or ''

        TextInput:
            id: text_input
            size_hint_y: None
            height: 30
            multiline: False

        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: "Cancel"
                on_release: root.cancel()

            Button:
                text: "Save"
                on_release: root.save(filechooser.path, text_input.text)
''')

################# RECYCLEVIEW ################### 

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    ''' Adds selection and focus behaviour to the view. '''

class SelectableLabel(RecycleDataViewBehavior, Label):
    ''' Add selection support to the Label '''
    index       = None
    selected    = BooleanProperty(False)
    selectable  = BooleanProperty(True)

    talon       = BooleanProperty(False)
    puntera     = BooleanProperty(False)

    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index  = index
        
        if rv.data[index]['talon'] == 1:
            self.talon = BooleanProperty(True)
        if rv.data[index]['talon'] == 0:
            self.talon = BooleanProperty(False)
        if rv.data[index]['puntera'] == 1:
            self.puntera = BooleanProperty(True)
        if rv.data[index]['puntera'] == 0:
            self.puntera = BooleanProperty(False)

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

            self.parent.parent.parent.grid_buttons_imgs.images.index_img = int(rv.data[index]['text'])
            self.parent.parent.parent.grid_buttons_imgs.images.name_img  = 'frame_{}.jpg'.format(self.parent.parent.parent.grid_buttons_imgs.images.index_img)

            self.parent.parent.parent.grid_buttons_imgs.images.remove_widget(self.parent.parent.parent.grid_buttons_imgs.images.actual_img)

            self.parent.parent.parent.grid_buttons_imgs.images.actual_img = Image(source="{}/{}".format(self.parent.parent.parent.grid_buttons_imgs.images.route, self.parent.parent.parent.grid_buttons_imgs.images.name_img))
            self.parent.parent.parent.grid_buttons_imgs.images.add_widget(self.parent.parent.parent.grid_buttons_imgs.images.actual_img)

        else:
            print("selection removed for {} ()".format(rv.data[index]['name']))

class frames_recycleview(RecycleView):
        def __init__(self, route, **kwargs):
            super(frames_recycleview, self).__init__(**kwargs)
            frames_list = self.sort_frames_list(self.ls(route))
            self.data = [{'text': str(x[0]), 'name': x[1], 'talon': 0, 'puntera': 0} for x in frames_list]

        def ls(self, ruta = getcwd()):
            return [arch.name for arch in scandir(ruta) if arch.is_file()]

        def sort_frames_list(self, frames_list):
            crop_name = lambda x: (int(x[6:].split('.')[0]), x)
            list_def = []
            
            for element in frames_list:
                list_def.append(crop_name(element))

            list_def.sort(key=lambda tup: tup[0])

            return list_def

################# BUTTONS ################### 

class MyButton(ButtonBehavior, Image):
    def __init__(self, route,**kwargs):
        super(MyButton, self).__init__(**kwargs)
        self.route  = route
        self.source = route

    def on_press(self):
        self.source = self.route
    def on_release(self):
        self.source = self.route

class button_container_img(GridLayout):
    def __init__(self, icon_route, **kwargs):
        super(button_container_img, self).__init__(**kwargs)
        self.cols = 6
        
        self.button_next            = MyButton("{}/next.png".format(icon_route))
        self.button_prev            = MyButton("{}/previous.png".format(icon_route))
        self.button_mark_talon      = MyButton("{}/mark.png".format(icon_route))
        self.button_mark_puntera    = MyButton("{}/mark.png".format(icon_route))
        self.button_dir             = MyButton("{}/dir.png".format(icon_route))
        self.button_save            = MyButton("{}/save.png".format(icon_route)) 

        self.add_widget(self.button_prev)
        self.add_widget(self.button_next)
        self.add_widget(self.button_mark_talon)
        self.add_widget(self.button_mark_puntera)
        self.add_widget(self.button_dir)
        self.add_widget(self.button_save)

################# IMAGES ################### 

class image_container(GridLayout):
    def __init__(self, route, **kwargs):
        super().__init__(**kwargs)
        self.cols       = 1

        self.index_img  = 0 
        self.name_img   = 'frame_0.jpg'
        self.route      = route

        self.actual_img = Image(source = "{}/frame_0.jpg".format(route))
        self.add_widget(self.actual_img)

################# BUTTONS AND IMAGES ################### 

class buttons_and_imgs_cont(GridLayout):
    def __init__(self, route_dir, route_icon, **kwargs):
        super(buttons_and_imgs_cont, self).__init__(**kwargs)
        self.cols = 1
        
        self.images = image_container(route=route_dir)        
        self.add_widget(self.images)

        self.button_bar = button_container_img(icon_route=route_icon)
        self.button_bar.size_hint = (1, 0.2)

        self.add_widget(self.button_bar)

        self.button_bar.button_next.bind(on_press=self.advance_img)
        self.button_bar.button_prev.bind(on_press=self.retrocess_img)
        self.button_bar.button_dir.bind(on_press=self.choose_dir)
        self.button_bar.button_mark_talon.bind(on_press=self.mark_talon)
        self.button_bar.button_mark_puntera.bind(on_press=self.mark_puntera)
        self.button_bar.button_save.bind(on_press=self.show_save)

    def advance_img(self, instance):
        if self.images.index_img < len(self.parent.frames_list_rv.data) - 1:
            self.images.index_img += 1
            self.images.name_img  = 'frame_{}.jpg'.format(self.images.index_img)

            self.images.remove_widget(self.images.actual_img)

            self.images.actual_img = Image(source="{}/{}".format(self.images.route, self.images.name_img))
            self.images.add_widget(self.images.actual_img)

    def retrocess_img(self, instance):
        if self.images.index_img > 0:
            self.images.index_img -= 1
            self.images.name_img  = 'frame_{}.jpg'.format(self.images.index_img)

            self.images.remove_widget(self.images.actual_img)

            self.images.actual_img = Image(source="{}/{}".format(self.images.route, self.images.name_img))
            self.images.add_widget(self.images.actual_img)

    def mark_talon(self, instance):
        main_window = self.parent

        if not self.images.index_img in main_window.talon or len(main_window.talon) == 0:
            
            if not self.images.index_img in main_window.puntera:
                main_window.talon.append(self.images.index_img)
                main_window.frames_list_rv.data[self.images.index_img]['talon'] = 1
            else:
                main_window.talon.append(self.images.index_img)
                main_window.puntera.pop(main_window.puntera.index(self.images.index_img))

                main_window.frames_list_rv.data[self.images.index_img]['talon'] = 1
                main_window.frames_list_rv.data[self.images.index_img]['puntera'] = 0
        else:
            main_window.talon.pop(main_window.talon.index(self.images.index_img))
            main_window.frames_list_rv.data[self.images.index_img]['talon'] = 0

    def mark_puntera(self, instance):
        main_window = self.parent

        if not self.images.index_img in main_window.puntera or len(main_window.puntera) == 0:

            if not self.images.index_img in main_window.talon:
                main_window.puntera.append(self.images.index_img)
                main_window.frames_list_rv.data[self.images.index_img]['puntera'] = 1

            else:
                main_window.puntera.append(self.images.index_img)
                main_window.talon.pop(main_window.talon.index(self.images.index_img))

                main_window.frames_list_rv.data[self.images.index_img]['talon'] = 0
                main_window.frames_list_rv.data[self.images.index_img]['puntera'] = 1
        else:
            main_window.puntera.pop(main_window.puntera.index(self.images.index_img))
            main_window.frames_list_rv.data[self.images.index_img]['puntera'] = 0

    def choose_dir(self, instance):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
    
    def load(self, path, filename):
        frames_route = os.path.join(path, "")
        frames_route = frames_route[:len(frames_route)-1]
        main_window = self.parent

        main_window.route_frames = frames_route
        main_window.remove_widget(main_window.frames_list_rv)
        main_window.remove_widget(main_window.grid_buttons_imgs)

        main_window.frames_list_rv = frames_recycleview(route=frames_route)
        main_window.grid_buttons_imgs = buttons_and_imgs_cont(route_dir=main_window.route_frames, route_icon=routes.icons_route)

        print(main_window.frames_list_rv.data)
        main_window.add_widget(main_window.frames_list_rv)
        main_window.add_widget(main_window.grid_buttons_imgs)
        
        self.dismiss_popup()

    def show_save(self, instance):
        content = SaveDialog(save=self.save_marks, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
    
    def save_marks(self, path, instance):
        file_talon      = open(os.path.join(path, "labeled_talon_hit_{}.csv".format(time.time())), "w")
        file_puntera    = open(os.path.join(path, "labeled_puntera_hit_{}.csv".format(time.time())), "w")

        main_window = self.parent

        file_talon.write("Frame")
        file_puntera.write("Frame")

        for hit in main_window.talon:
            file_talon.write("{}\n".format(hit))

        for hit in main_window.puntera:
            file_puntera.write("{}\n".format(hit))

        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

################# MAINWINDOW ################### 

class MainWindow(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2
        self.route_frames = "{}/{}/frames".format(os.getcwd(), routes.icons_route)

        self.talon          = []
        self.puntera        = []
        self.actual_index   = 0

        self.frames_list_rv = frames_recycleview(route=self.route_frames)
        self.add_widget(self.frames_list_rv)

        self.grid_buttons_imgs = buttons_and_imgs_cont(route_dir=self.route_frames, route_icon=routes.icons_route)
        self.add_widget(self.grid_buttons_imgs)

############## FILEBROWSER ##################

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

################# EPICAPP ####################

class EpicApp(App):
    title           = "KNIGHT"
    frames_route    = ""
    build_window    = False

    def build(self):
        main_window = MainWindow()
        return main_window

################# MAIN ################### 

if __name__ == '__main__':
    EpicApp().run()

#############################################