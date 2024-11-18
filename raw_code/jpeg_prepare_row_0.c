static void jpeg_prepare_row(VncState *vs, uint8_t *dst, int x, int y,

                                       int count)

{

    if (vs->tight_pixel24)

        jpeg_prepare_row24(vs, dst, x, y, count);

    else if (ds_get_bytes_per_pixel(vs->ds) == 4)

        jpeg_prepare_row32(vs, dst, x, y, count);

    else

        jpeg_prepare_row16(vs, dst, x, y, count);

}
