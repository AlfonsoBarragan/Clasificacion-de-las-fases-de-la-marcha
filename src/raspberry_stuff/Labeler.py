# Interfaz y logica para etiquetar los datos manualmente
import tkinter as tk

root = tk.Tk()

# Setup
root.title("Labeller")
root.geometry("1280x720")

# Frames

frame_for_frames = tk.Frame(root)
frame_for_frames.pack(fill="both")
frame_for_frames.config(bg="red", width="640", height="360")

frame_for_actual_image = tk.Frame(root)
frame_for_actual_image.pack(fill="x", side="right")
frame_for_frames.config(bg="blue", width="320", height="180")

# Widgets


# Execution

root.mainloop()