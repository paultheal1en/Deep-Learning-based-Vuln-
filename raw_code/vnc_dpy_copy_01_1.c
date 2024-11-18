static void vnc_dpy_copy(DisplayState *ds, int src_x, int src_y, int dst_x, int dst_y, int w, int h)

{

    VncDisplay *vd = ds->opaque;

    VncState *vs = vd->clients;

    while (vs != NULL) {

        if (vnc_has_feature(vs, VNC_FEATURE_COPYRECT))

            vnc_copy(vs, src_x, src_y, dst_x, dst_y, w, h);

        else /* TODO */

            vnc_update(vs, dst_x, dst_y, w, h);

        vs = vs->next;

    }

}
