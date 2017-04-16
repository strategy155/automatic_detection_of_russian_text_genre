from msvcrt import getch, kbhit


def check_if_key_pressed(ord_code):
    if kbhit():
        if ord(getch()) == ord_code:
            return True

