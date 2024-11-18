static void vnc_refresh(DisplayChangeListener *dcl)

{

    VncDisplay *vd = container_of(dcl, VncDisplay, dcl);

    VncState *vs, *vn;

    int has_dirty, rects = 0;



    graphic_hw_update(NULL);



    if (vnc_trylock_display(vd)) {

        update_displaychangelistener(&vd->dcl, VNC_REFRESH_INTERVAL_BASE);

        return;

    }



    has_dirty = vnc_refresh_server_surface(vd);

    vnc_unlock_display(vd);



    QTAILQ_FOREACH_SAFE(vs, &vd->clients, next, vn) {

        rects += vnc_update_client(vs, has_dirty);

        /* vs might be free()ed here */

    }



    if (QTAILQ_EMPTY(&vd->clients)) {

        update_displaychangelistener(&vd->dcl, VNC_REFRESH_INTERVAL_MAX);

        return;

    }



    if (has_dirty && rects) {

        vd->dcl.update_interval /= 2;

        if (vd->dcl.update_interval < VNC_REFRESH_INTERVAL_BASE) {

            vd->dcl.update_interval = VNC_REFRESH_INTERVAL_BASE;

        }

    } else {

        vd->dcl.update_interval += VNC_REFRESH_INTERVAL_INC;

        if (vd->dcl.update_interval > VNC_REFRESH_INTERVAL_MAX) {

            vd->dcl.update_interval = VNC_REFRESH_INTERVAL_MAX;

        }

    }

}
