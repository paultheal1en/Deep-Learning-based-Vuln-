static gboolean gd_motion_event(GtkWidget *widget, GdkEventMotion *motion,

                                void *opaque)

{

    GtkDisplayState *s = opaque;

    int dx, dy;

    int x, y;



    x = motion->x / s->scale_x;

    y = motion->y / s->scale_y;



    if (kbd_mouse_is_absolute()) {

        dx = x * 0x7FFF / (ds_get_width(s->ds) - 1);

        dy = y * 0x7FFF / (ds_get_height(s->ds) - 1);

    } else if (s->last_x == -1 || s->last_y == -1) {

        dx = 0;

        dy = 0;

    } else {

        dx = x - s->last_x;

        dy = y - s->last_y;

    }



    s->last_x = x;

    s->last_y = y;



    if (kbd_mouse_is_absolute()) {

        kbd_mouse_event(dx, dy, 0, s->button_mask);

    }



    return TRUE;

}
