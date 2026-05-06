fixed = open('main.cc').read()
fixed = fixed.replace('UART_BR_115200', 'UART_BR_921600')
open('main.cc', 'w').write(fixed)
print("Baud rate updated to 921600")
