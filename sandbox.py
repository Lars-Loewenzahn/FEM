import sys
import os

def check_tcl_tk_installation():
    try:
        import tkinter
        tcl_path = tkinter.Tcl().eval('info library')
        tk_path = tkinter.Tk().tk.eval('info library')
        
        print("Tkinter erfolgreich importiert.")
        print(f"Tcl Bibliothek Pfad: {tcl_path}")
        print(f"Tk Bibliothek Pfad: {tk_path}")
        
        # Überprüfen, ob die Pfade existieren
        if os.path.exists(tcl_path) and os.path.exists(tk_path):
            print("Tcl und Tk sind korrekt installiert.")
        else:
            print("Tcl oder Tk sind nicht korrekt installiert.")
    
    except ImportError:
        print("Tkinter konnte nicht importiert werden. Tcl/Tk ist möglicherweise nicht installiert.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    check_tcl_tk_installation()