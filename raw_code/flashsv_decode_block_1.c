static int flashsv_decode_block(AVCodecContext *avctx, AVPacket *avpkt,

                                GetBitContext *gb, int block_size,

                                int width, int height, int x_pos, int y_pos,

                                int blk_idx)

{

    struct FlashSVContext *s = avctx->priv_data;

    uint8_t *line = s->tmpblock;

    int k;

    int ret = inflateReset(&s->zstream);

    if (ret != Z_OK) {

        av_log(avctx, AV_LOG_ERROR, "Inflate reset error: %d\n", ret);

        return AVERROR_UNKNOWN;

    }

    if (s->zlibprime_curr || s->zlibprime_prev) {

        ret = flashsv2_prime(s,

                             s->blocks[blk_idx].pos,

                             s->blocks[blk_idx].size);

        if (ret < 0)

            return ret;

    }

    s->zstream.next_in   = avpkt->data + get_bits_count(gb) / 8;

    s->zstream.avail_in  = block_size;

    s->zstream.next_out  = s->tmpblock;

    s->zstream.avail_out = s->block_size * 3;

    ret = inflate(&s->zstream, Z_FINISH);

    if (ret == Z_DATA_ERROR) {

        av_log(avctx, AV_LOG_ERROR, "Zlib resync occurred\n");

        inflateSync(&s->zstream);

        ret = inflate(&s->zstream, Z_FINISH);

    }



    if (ret != Z_OK && ret != Z_STREAM_END) {

        //return -1;

    }



    if (s->is_keyframe) {

        s->blocks[blk_idx].pos  = s->keyframedata + (get_bits_count(gb) / 8);

        s->blocks[blk_idx].size = block_size;

    }



    y_pos += s->diff_start;



    if (!s->color_depth) {

        /* Flash Screen Video stores the image upside down, so copy

         * lines to destination in reverse order. */

        for (k = 1; k <= s->diff_height; k++) {

            memcpy(s->frame->data[0] + x_pos * 3 +

                   (s->image_height - y_pos - k) * s->frame->linesize[0],

                   line, width * 3);

            /* advance source pointer to next line */

            line += width * 3;

        }

    } else {

        /* hybrid 15-bit/palette mode */

        decode_hybrid(s->tmpblock, s->frame->data[0],

                      s->image_height - (y_pos + 1 + s->diff_height),

                      x_pos, s->diff_height, width,

                      s->frame->linesize[0], s->pal);

    }

    skip_bits_long(gb, 8 * block_size); /* skip the consumed bits */

    return 0;

}
