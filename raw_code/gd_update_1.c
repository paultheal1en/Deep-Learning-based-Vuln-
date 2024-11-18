static void gd_update(DisplayChangeListener *dcl,

                      DisplayState *ds, int x, int y, int w, int h)

{

    GtkDisplayState *s = ds->opaque;

    int x1, x2, y1, y2;

    int mx, my;

    int fbw, fbh;

    int ww, wh;



    DPRINTF("update(x=%d, y=%d, w=%d, h=%d)\n", x, y, w, h);



    x1 = floor(x * s->scale_x);

    y1 = floor(y * s->scale_y);



    x2 = ceil(x * s->scale_x + w * s->scale_x);

    y2 = ceil(y * s->scale_y + h * s->scale_y);



    fbw = ds_get_width(s->ds) * s->scale_x;

    fbh = ds_get_height(s->ds) * s->scale_y;



    gdk_drawable_get_size(gtk_widget_get_window(s->drawing_area), &ww, &wh);



    mx = my = 0;

    if (ww > fbw) {

        mx = (ww - fbw) / 2;

    }

    if (wh > fbh) {

        my = (wh - fbh) / 2;

    }



    gtk_widget_queue_draw_area(s->drawing_area, mx + x1, my + y1, (x2 - x1), (y2 - y1));

}
