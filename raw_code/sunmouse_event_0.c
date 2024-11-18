static void sunmouse_event(void *opaque,

                               int dx, int dy, int dz, int buttons_state)

{

    ChannelState *s = opaque;

    int ch;



    MS_DPRINTF("dx=%d dy=%d buttons=%01x\n", dx, dy, buttons_state);



    ch = 0x80 | 0x7; /* protocol start byte, no buttons pressed */



    if (buttons_state & MOUSE_EVENT_LBUTTON)

        ch ^= 0x4;

    if (buttons_state & MOUSE_EVENT_MBUTTON)

        ch ^= 0x2;

    if (buttons_state & MOUSE_EVENT_RBUTTON)

        ch ^= 0x1;



    put_queue(s, ch);



    ch = dx;



    if (ch > 127)

        ch=127;

    else if (ch < -127)

        ch=-127;



    put_queue(s, ch & 0xff);



    ch = -dy;



    if (ch > 127)

        ch=127;

    else if (ch < -127)

        ch=-127;



    put_queue(s, ch & 0xff);



    // MSC protocol specify two extra motion bytes



    put_queue(s, 0);

    put_queue(s, 0);

}
