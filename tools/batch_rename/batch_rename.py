from tkinter import Tk, Label, Button, Text, StringVar, Frame, Checkbutton, BooleanVar
from tkinter.filedialog import askdirectory
import os
from datetime import date, datetime

class GUI:

    def rename_btn_cmd(self):

        success = self.rename()
        if success:
            self.update_founded_files()


    def open_selected_folder_btn_cmd(self):

        self.open_select_folder()
        self.update_founded_files()


    def rename(self):

        index= 1

        new_name_prefix = self.file_name_text.get("1.0",'end-1c')

        if(new_name_prefix == ""):
            self.error.set("Empty name")
            return False

        if self.path.get() != "":

            file_list = os.listdir(self.path.get())

            if len(file_list) != 0:
                    
                for filename in file_list:
                    
                    # get extension
                    filename_split_dot = filename.split(".")
                    extension = filename_split_dot[len(filename_split_dot)-1]

                    options = ""

                    if not self.use_index.get() and not self.use_time.get():

                        self.error.set("Names must be differents, please use time or index")
                        return False

                    if self.use_index.get():
                        options = "_" + str(index)

                    if self.use_date.get():
                        options = options + "_" + str(date.today())

                    if self.use_time.get():
                        options = options + "_" + str(datetime.now().time())
                    
                    new_filename = f"{new_name_prefix}{options}.{extension}"
                    
                    current_file = self.path.get() + "/" +  filename
                    new_file = self.path.get() + "/" + new_filename

                    os.rename(current_file, new_file)
                    # print(name + "." + extension)

                    index = index + 1

                self.error.set("")

            else:
                self.error.set("No file detected")
                return False

        else:
            self.error.set("No folder selected")

            return False
        
        return True


    def update_founded_files(self):

        files_founded = ''
        
        add_separation = False
        if self.path.get() != "":
            for filename in os.listdir(self.path.get()):
                
                if add_separation:
                    files_founded = files_founded + ", " + filename
                else:
                    files_founded = filename
                    add_separation = True

            files_founded = files_founded[:500] 

            self.files.set(files_founded)





    def open_select_folder(self):

        path_value = askdirectory(title='Select Folder')
        self.path.set(path_value)

        self.update_founded_files()
        

    def build_window(self):


        self.root.title("Batch rename")
        self.root.minsize(800, 100)  # width, height
        self.root.geometry("300x300+50+50")


    def build_select_folder_frame(self):

        self.select_folder_frame = Frame(self.root)

        # path label
        path_label = Label(self.select_folder_frame, text="Path: ")
        path_label.pack(side='left')

        # path value
        path_value = Label(self.select_folder_frame, textvariable=self.path, padx=10)
        path_value.pack(side='left')
        
        # select folder btn
        select_folder_btn = Button(self.select_folder_frame, text="Select folder", command=lambda *args: self.open_selected_folder_btn_cmd())
        select_folder_btn.pack(side='right')

        self.select_folder_frame.pack(pady=10)


    def build_file_name_frame(self):

        self.file_name_frame = Frame(self.root)

        # file name label
        file_name_label = Label(self.file_name_frame, text="File name   ")
        file_name_label.pack(side='left')

        # file name text
        self.file_name_text = Text(self.file_name_frame, height=1)
        self.file_name_text.pack(side='left')

        self.file_name_frame.pack(pady=10)


    def build_options_frame(self):

        self.options_frame = Frame(self.root)

        use_index_btn = Checkbutton(self.options_frame, text="Use index: ", variable=self.use_index).pack(side="left")
        use_date_btn = Checkbutton(self.options_frame, text="Use date: ", variable=self.use_date).pack(side="left")
        use_time_btn = Checkbutton(self.options_frame, text="Use time: ", variable=self.use_time).pack(side="left")

        
        self.options_frame.pack(pady=10)

    def build_rename_frame(self):

        self.rename_frame = Frame(self.root)

        # rename btn
        rename_files_btn = Button(self.rename_frame, text="Rename", command=lambda *args: self.rename_btn_cmd())
        rename_files_btn.pack(side='left')

        # files label
        files_label = Label(self.rename_frame, text="Files: ")
        files_label.pack(side='left')

        # files names
        filenames_label = Label(self.rename_frame, textvariable=self.files)
        filenames_label.pack(side='bottom')

        
        self.rename_frame.pack(pady=10)

    
    def build_error_frame(self):

        self.error_frame = Frame(self.root)

        # files names
        error_label = Label(self.error_frame, textvariable=self.error, fg="#F00")
        error_label.pack(side='bottom')

        
        self.error_frame.pack(pady=10, side="bottom")




    def __init__(self):
            
        self.root = Tk()

        self.path = StringVar(value="")
        self.files = StringVar(value="")
        self.error = StringVar(value="")
        self.use_date = BooleanVar(value=False)
        self.use_time = BooleanVar(value=False)
        self.use_index = BooleanVar(value=False)
        self.use_index.set(True)

        self.build_window()
        self.build_select_folder_frame()
        self.build_file_name_frame()
        self.build_options_frame()
        self.build_rename_frame()
        self.build_error_frame()
        
        self.root.mainloop()

gui = GUI()