static int gif_read_image(GifState *s)

{

    int left, top, width, height, bits_per_pixel, code_size, flags;

    int is_interleaved, has_local_palette, y, pass, y1, linesize, n, i;

    uint8_t *ptr, *spal, *palette, *ptr1;



    left = bytestream_get_le16(&s->bytestream);

    top = bytestream_get_le16(&s->bytestream);

    width = bytestream_get_le16(&s->bytestream);

    height = bytestream_get_le16(&s->bytestream);

    flags = bytestream_get_byte(&s->bytestream);

    is_interleaved = flags & 0x40;

    has_local_palette = flags & 0x80;

    bits_per_pixel = (flags & 0x07) + 1;



    av_dlog(s->avctx, "image x=%d y=%d w=%d h=%d\n", left, top, width, height);



    if (has_local_palette) {

        bytestream_get_buffer(&s->bytestream, s->local_palette, 3 * (1 << bits_per_pixel));

        palette = s->local_palette;

    } else {

        palette = s->global_palette;

        bits_per_pixel = s->bits_per_pixel;

    }



    /* verify that all the image is inside the screen dimensions */

    if (left + width > s->screen_width ||

        top + height > s->screen_height)

        return AVERROR(EINVAL);



    /* build the palette */

    n = (1 << bits_per_pixel);

    spal = palette;

    for(i = 0; i < n; i++) {

        s->image_palette[i] = (0xffu << 24) | AV_RB24(spal);

        spal += 3;

    }

    for(; i < 256; i++)

        s->image_palette[i] = (0xffu << 24);

    /* handle transparency */

    if (s->transparent_color_index >= 0)

        s->image_palette[s->transparent_color_index] = 0;



    /* now get the image data */

    code_size = bytestream_get_byte(&s->bytestream);

    ff_lzw_decode_init(s->lzw, code_size, s->bytestream,

                       s->bytestream_end - s->bytestream, FF_LZW_GIF);



    /* read all the image */

    linesize = s->picture.linesize[0];

    ptr1 = s->picture.data[0] + top * linesize + left;

    ptr = ptr1;

    pass = 0;

    y1 = 0;

    for (y = 0; y < height; y++) {

        ff_lzw_decode(s->lzw, ptr, width);

        if (is_interleaved) {

            switch(pass) {

            default:

            case 0:

            case 1:

                y1 += 8;

                ptr += linesize * 8;

                if (y1 >= height) {

                    y1 = pass ? 2 : 4;

                    ptr = ptr1 + linesize * y1;

                    pass++;

                }

                break;

            case 2:

                y1 += 4;

                ptr += linesize * 4;

                if (y1 >= height) {

                    y1 = 1;

                    ptr = ptr1 + linesize;

                    pass++;

                }

                break;

            case 3:

                y1 += 2;

                ptr += linesize * 2;

                break;

            }

        } else {

            ptr += linesize;

        }

    }

    /* read the garbage data until end marker is found */

    ff_lzw_decode_tail(s->lzw);

    s->bytestream = ff_lzw_cur_ptr(s->lzw);

    return 0;

}
