static void vnc_refresh(void *opaque)

{

    VncDisplay *vd = opaque;

    VncState *vs, *vn;

    int has_dirty, rects = 0;



    vga_hw_update();



    if (vnc_trylock_display(vd)) {

        vd->timer_interval = VNC_REFRESH_INTERVAL_BASE;

        qemu_mod_timer(vd->timer, qemu_get_clock(rt_clock) +

                       vd->timer_interval);

        return;

    }



    has_dirty = vnc_refresh_server_surface(vd);

    vnc_unlock_display(vd);



    QTAILQ_FOREACH_SAFE(vs, &vd->clients, next, vn) {

        rects += vnc_update_client(vs, has_dirty);

        /* vs might be free()ed here */

    }



    /* vd->timer could be NULL now if the last client disconnected,

     * in this case don't update the timer */

    if (vd->timer == NULL)

        return;



    if (has_dirty && rects) {

        vd->timer_interval /= 2;

        if (vd->timer_interval < VNC_REFRESH_INTERVAL_BASE)

            vd->timer_interval = VNC_REFRESH_INTERVAL_BASE;

    } else {

        vd->timer_interval += VNC_REFRESH_INTERVAL_INC;

        if (vd->timer_interval > VNC_REFRESH_INTERVAL_MAX)

            vd->timer_interval = VNC_REFRESH_INTERVAL_MAX;

    }

    qemu_mod_timer(vd->timer, qemu_get_clock(rt_clock) + vd->timer_interval);

}
