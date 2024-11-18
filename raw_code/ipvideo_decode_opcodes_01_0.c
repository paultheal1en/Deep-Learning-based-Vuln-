static void ipvideo_decode_opcodes(IpvideoContext *s)

{

    int x, y;

    unsigned char opcode;

    int ret;

    static int frame = 0;

    GetBitContext gb;



    debug_interplay("------------------ frame %d\n", frame);

    frame++;



    /* this is PAL8, so make the palette available */

    memcpy(s->current_frame.data[1], s->avctx->palctrl->palette, PALETTE_COUNT * 4);



    s->stride = s->current_frame.linesize[0];

    s->stream_ptr = s->buf + 14;  /* data starts 14 bytes in */

    s->stream_end = s->buf + s->size;

    s->line_inc = s->stride - 8;

    s->upper_motion_limit_offset = (s->avctx->height - 8) * s->stride

        + s->avctx->width - 8;



    init_get_bits(&gb, s->decoding_map, s->decoding_map_size * 8);

    for (y = 0; y < (s->stride * s->avctx->height); y += s->stride * 8) {

        for (x = y; x < y + s->avctx->width; x += 8) {

            opcode = get_bits(&gb, 4);



            debug_interplay("  block @ (%3d, %3d): encoding 0x%X, data ptr @ %p\n",

                            x - y, y / s->stride, opcode, s->stream_ptr);



            s->pixel_ptr = s->current_frame.data[0] + x;

            ret = ipvideo_decode_block[opcode](s);

            if (ret != 0) {

                av_log(s->avctx, AV_LOG_ERROR, " Interplay video: decode problem on frame %d, @ block (%d, %d)\n",

                       frame, x - y, y / s->stride);

                return;

            }

        }

    }

    if (s->stream_end - s->stream_ptr > 1) {

        av_log(s->avctx, AV_LOG_ERROR, " Interplay video: decode finished with %td bytes left over\n",

               s->stream_end - s->stream_ptr);

    }

}
