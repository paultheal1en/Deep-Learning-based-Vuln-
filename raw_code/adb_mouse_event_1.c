static void adb_mouse_event(void *opaque,

                            int dx1, int dy1, int dz1, int buttons_state)

{

    MouseState *s = opaque;



    s->dx += dx1;

    s->dy += dy1;

    s->dz += dz1;

    s->buttons_state = buttons_state;

}
