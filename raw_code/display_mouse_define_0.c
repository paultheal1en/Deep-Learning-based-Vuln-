static void display_mouse_define(DisplayChangeListener *dcl,

                                 QEMUCursor *c)

{

    SimpleSpiceDisplay *ssd = container_of(dcl, SimpleSpiceDisplay, dcl);



    qemu_mutex_lock(&ssd->lock);

    if (c) {

        cursor_get(c);

    }

    cursor_put(ssd->cursor);

    ssd->cursor = c;

    ssd->hot_x = c->hot_x;

    ssd->hot_y = c->hot_y;

    g_free(ssd->ptr_move);

    ssd->ptr_move = NULL;

    g_free(ssd->ptr_define);

    ssd->ptr_define = qemu_spice_create_cursor_update(ssd, c, 0);

    qemu_mutex_unlock(&ssd->lock);

}
