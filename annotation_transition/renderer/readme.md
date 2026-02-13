Mouse handler -> call policy and render engine, changes renderer draw state



mouse_handler -> policy, action_handler


falta fazer o policy para tratar o toggle entre o start e o done da mask
o problema é que se for create, deve apenas mudar o draw state e o polygon
se for done, deve mandar para fora com o dispatcher