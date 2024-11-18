static void vnc_init_timer(VncDisplay *vd)

{

    vd->timer_interval = VNC_REFRESH_INTERVAL_BASE;

    if (vd->timer == NULL && !QTAILQ_EMPTY(&vd->clients)) {

        vd->timer = qemu_new_timer_ms(rt_clock, vnc_refresh, vd);

        vnc_dpy_resize(dcl, vd->ds);

        vnc_refresh(vd);

    }

}
