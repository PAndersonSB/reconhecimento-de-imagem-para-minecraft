import numpy as np
import mss
import mss.tools


class WindowCaptureLinux:

    # properties
    w = 0
    h = 0
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name="", size=(1900, 1000), origin=(0, 0)):
        self.size = size
        self.origin = origin 

        # Para Linux, assumimos captura de tela completa ou de uma região específica
        self.window_name = window_name
        if window_name == "":
            self.w = self.size[0]
            self.h = self.size[1]
        else:
            raise NotImplementedError("Detecção de janelas específicas não foi implementada para Linux")

    def get_screenshot(self):
        # Use mss para capturar a tela ou uma região
        with mss.mss() as sct:
            #print(sct.monitors)
            # Define a área de captura
            monitor = {
                "top": self.origin[1],
                "left": self.origin[0],
                "width": self.w,
                "height": self.h,
            }
            
            # Captura a região definida
            screenshot = sct.grab(monitor)

            # Converte a captura para um array numpy
            img = np.array(screenshot)

            # Remove o canal alfa
            img = img[..., :3]

            # Torna a imagem C_CONTIGUOUS para evitar erros
            img = np.ascontiguousarray(img)

        return img

    # Para listar janelas em Linux, você precisaria usar bibliotecas como Xlib
    def list_window_names(self):
        raise NotImplementedError("Listagem de janelas não foi implementada para Linux")

    def get_screen_position(self, pos):
        # Simplesmente retorna a posição com base na origem
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)
