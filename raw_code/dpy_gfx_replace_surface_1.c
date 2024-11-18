void dpy_gfx_replace_surface(QemuConsole *con,
                             DisplaySurface *surface)
{
    DisplayState *s = con->ds;
    DisplaySurface *old_surface = con->surface;
    DisplayChangeListener *dcl;
    con->surface = surface;
    QLIST_FOREACH(dcl, &s->listeners, next) {
        if (con != (dcl->con ? dcl->con : active_console)) {
            continue;
        }
        if (dcl->ops->dpy_gfx_switch) {
            dcl->ops->dpy_gfx_switch(dcl, surface);
        }
    }
    qemu_free_displaysurface(old_surface);
}