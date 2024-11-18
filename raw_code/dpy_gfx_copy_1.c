void dpy_gfx_copy(QemuConsole *con, int src_x, int src_y,

                  int dst_x, int dst_y, int w, int h)

{

    DisplayState *s = con->ds;

    DisplayChangeListener *dcl;



    if (!qemu_console_is_visible(con)) {

        return;

    }

    QLIST_FOREACH(dcl, &s->listeners, next) {

        if (con != (dcl->con ? dcl->con : active_console)) {

            continue;

        }

        if (dcl->ops->dpy_gfx_copy) {

            dcl->ops->dpy_gfx_copy(dcl, src_x, src_y, dst_x, dst_y, w, h);

        } else { /* TODO */

            dcl->ops->dpy_gfx_update(dcl, dst_x, dst_y, w, h);

        }

    }

}
