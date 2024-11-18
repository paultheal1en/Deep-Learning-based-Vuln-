static void send_framebuffer_update_raw(VncState *vs, int x, int y, int w, int h)

{

    int i;

    uint8_t *row;



    row = ds_get_data(vs->ds) + y * ds_get_linesize(vs->ds) + x * ds_get_bytes_per_pixel(vs->ds);

    for (i = 0; i < h; i++) {

        vs->write_pixels(vs, row, w * ds_get_bytes_per_pixel(vs->ds));

        row += ds_get_linesize(vs->ds);

    }

}
