static int gif_read_image(GifState *s, AVFrame *frame)

{

    int left, top, width, height, bits_per_pixel, code_size, flags;

    int is_interleaved, has_local_palette, y, pass, y1, linesize, pal_size;

    uint32_t *ptr, *pal, *px, *pr, *ptr1;

    int ret;

    uint8_t *idx;



    /* At least 9 bytes of Image Descriptor. */

    if (bytestream2_get_bytes_left(&s->gb) < 9)

        return AVERROR_INVALIDDATA;



    left   = bytestream2_get_le16u(&s->gb);

    top    = bytestream2_get_le16u(&s->gb);

    width  = bytestream2_get_le16u(&s->gb);

    height = bytestream2_get_le16u(&s->gb);

    flags  = bytestream2_get_byteu(&s->gb);

    is_interleaved = flags & 0x40;

    has_local_palette = flags & 0x80;

    bits_per_pixel = (flags & 0x07) + 1;



    av_dlog(s->avctx, "image x=%d y=%d w=%d h=%d\n", left, top, width, height);



    if (has_local_palette) {

        pal_size = 1 << bits_per_pixel;



        if (bytestream2_get_bytes_left(&s->gb) < pal_size * 3)

            return AVERROR_INVALIDDATA;



        gif_read_palette(s, s->local_palette, pal_size);

        pal = s->local_palette;

    } else {

        if (!s->has_global_palette) {

            av_log(s->avctx, AV_LOG_ERROR, "picture doesn't have either global or local palette.\n");

            return AVERROR_INVALIDDATA;

        }



        pal = s->global_palette;

    }



    if (s->keyframe) {

        if (s->transparent_color_index == -1 && s->has_global_palette) {

            /* transparency wasn't set before the first frame, fill with background color */

            gif_fill(frame, s->bg_color);

        } else {

            /* otherwise fill with transparent color.

             * this is necessary since by default picture filled with 0x80808080. */

            gif_fill(frame, s->trans_color);

        }

    }



    /* verify that all the image is inside the screen dimensions */

    if (left + width > s->screen_width ||

        top + height > s->screen_height) {

        av_log(s->avctx, AV_LOG_ERROR, "image is outside the screen dimensions.\n");

        return AVERROR_INVALIDDATA;

    }

    if (width <= 0 || height <= 0) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid image dimensions.\n");

        return AVERROR_INVALIDDATA;

    }



    /* process disposal method */

    if (s->gce_prev_disposal == GCE_DISPOSAL_BACKGROUND) {

        gif_fill_rect(frame, s->stored_bg_color, s->gce_l, s->gce_t, s->gce_w, s->gce_h);

    } else if (s->gce_prev_disposal == GCE_DISPOSAL_RESTORE) {

        gif_copy_img_rect(s->stored_img, (uint32_t *)frame->data[0],

            frame->linesize[0] / sizeof(uint32_t), s->gce_l, s->gce_t, s->gce_w, s->gce_h);

    }



    s->gce_prev_disposal = s->gce_disposal;



    if (s->gce_disposal != GCE_DISPOSAL_NONE) {

        s->gce_l = left;  s->gce_t = top;

        s->gce_w = width; s->gce_h = height;



        if (s->gce_disposal == GCE_DISPOSAL_BACKGROUND) {

            if (s->transparent_color_index >= 0)

                s->stored_bg_color = s->trans_color;

            else

                s->stored_bg_color = s->bg_color;

        } else if (s->gce_disposal == GCE_DISPOSAL_RESTORE) {

            av_fast_malloc(&s->stored_img, &s->stored_img_size, frame->linesize[0] * frame->height);

            if (!s->stored_img)

                return AVERROR(ENOMEM);



            gif_copy_img_rect((uint32_t *)frame->data[0], s->stored_img,

                frame->linesize[0] / sizeof(uint32_t), left, top, width, height);

        }

    }



    /* Expect at least 2 bytes: 1 for lzw code size and 1 for block size. */

    if (bytestream2_get_bytes_left(&s->gb) < 2)

        return AVERROR_INVALIDDATA;



    /* now get the image data */

    code_size = bytestream2_get_byteu(&s->gb);

    if ((ret = ff_lzw_decode_init(s->lzw, code_size, s->gb.buffer,

                                  bytestream2_get_bytes_left(&s->gb), FF_LZW_GIF)) < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "LZW init failed\n");

        return ret;

    }



    /* read all the image */

    linesize = frame->linesize[0] / sizeof(uint32_t);

    ptr1 = (uint32_t *)frame->data[0] + top * linesize + left;

    ptr = ptr1;

    pass = 0;

    y1 = 0;

    for (y = 0; y < height; y++) {

        int count = ff_lzw_decode(s->lzw, s->idx_line, width);

        if (count != width) {

            if (count)

                av_log(s->avctx, AV_LOG_ERROR, "LZW decode failed\n");

            goto decode_tail;

        }



        pr = ptr + width;



        for (px = ptr, idx = s->idx_line; px < pr; px++, idx++) {

            if (*idx != s->transparent_color_index)

                *px = pal[*idx];

        }



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



 decode_tail:

    /* read the garbage data until end marker is found */

    ff_lzw_decode_tail(s->lzw);



    /* Graphic Control Extension's scope is single frame.

     * Remove its influence. */

    s->transparent_color_index = -1;

    s->gce_disposal = GCE_DISPOSAL_NONE;



    return 0;

}
