static int gif_read_header1(GifState *s)

{

    ByteIOContext *f = s->f;

    uint8_t sig[6];

    int ret, v, n;

    int has_global_palette;



    /* read gif signature */

    ret = get_buffer(f, sig, 6);

    if (ret != 6)

        return -1;

    if (memcmp(sig, gif87a_sig, 6) != 0 &&

        memcmp(sig, gif89a_sig, 6) != 0)

        return -1;



    /* read screen header */

    s->transparent_color_index = -1;

    s->screen_width = get_le16(f);

    s->screen_height = get_le16(f);

    if(   (unsigned)s->screen_width  > 32767

       || (unsigned)s->screen_height > 32767){

        av_log(NULL, AV_LOG_ERROR, "picture size too large\n");

        return -1;

    }



    v = get_byte(f);

    s->color_resolution = ((v & 0x70) >> 4) + 1;

    has_global_palette = (v & 0x80);

    s->bits_per_pixel = (v & 0x07) + 1;

    s->background_color_index = get_byte(f);

    get_byte(f);                /* ignored */

#ifdef DEBUG

    printf("gif: screen_w=%d screen_h=%d bpp=%d global_palette=%d\n",

           s->screen_width, s->screen_height, s->bits_per_pixel,

           has_global_palette);

#endif

    if (has_global_palette) {

        n = 1 << s->bits_per_pixel;

        get_buffer(f, s->global_palette, n * 3);

    }

    return 0;

}
