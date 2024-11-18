static void vnc_copy(VncState *vs, int src_x, int src_y, int dst_x, int dst_y, int w, int h)

{

    /* send bitblit op to the vnc client */

    vnc_lock_output(vs);

    vnc_write_u8(vs, VNC_MSG_SERVER_FRAMEBUFFER_UPDATE);

    vnc_write_u8(vs, 0);

    vnc_write_u16(vs, 1); /* number of rects */

    vnc_framebuffer_update(vs, dst_x, dst_y, w, h, VNC_ENCODING_COPYRECT);

    vnc_write_u16(vs, src_x);

    vnc_write_u16(vs, src_y);

    vnc_unlock_output(vs);

    vnc_flush(vs);

}
