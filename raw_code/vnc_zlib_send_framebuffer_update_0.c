int vnc_zlib_send_framebuffer_update(VncState *vs, int x, int y, int w, int h)

{

    int old_offset, new_offset, bytes_written;



    vnc_framebuffer_update(vs, x, y, w, h, VNC_ENCODING_ZLIB);



    // remember where we put in the follow-up size

    old_offset = vs->output.offset;

    vnc_write_s32(vs, 0);



    // compress the stream

    vnc_zlib_start(vs);

    vnc_raw_send_framebuffer_update(vs, x, y, w, h);

    bytes_written = vnc_zlib_stop(vs);



    if (bytes_written == -1)

        return 0;



    // hack in the size

    new_offset = vs->output.offset;

    vs->output.offset = old_offset;

    vnc_write_u32(vs, bytes_written);

    vs->output.offset = new_offset;



    return 1;

}
