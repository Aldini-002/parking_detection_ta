import tkinter as tk
from function.setup import setup
from function.detection import detection

# GUI dengan tkinter
def create_gui():
    root = tk.Tk()
    root.title("Pilihan Kode")

    label = tk.Label(root, text="Pilih kode yang ingin dijalankan:")
    label.pack(padx=10, pady=10)

    button1 = tk.Button(root, text="Setup Parkir", command=setup)
    button1.pack(padx=10, pady=10)

    button2 = tk.Button(root, text="Deteksi parkir", command=detection)
    button2.pack(padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
