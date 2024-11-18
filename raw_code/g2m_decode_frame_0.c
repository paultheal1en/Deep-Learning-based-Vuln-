static int g2m_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_picture_ptr, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    G2MContext *c = avctx->priv_data;

    AVFrame *pic = data;

    GetByteContext bc, tbc;

    int magic;

    int got_header = 0;

    uint32_t chunk_size, cur_size;

    int chunk_type;

    int i;

    int ret;



    if (buf_size < 12) {

        av_log(avctx, AV_LOG_ERROR,

               "Frame should have at least 12 bytes, got %d instead\n",

               buf_size);

        return AVERROR_INVALIDDATA;

    }



    bytestream2_init(&bc, buf, buf_size);



    magic = bytestream2_get_be32(&bc);

    if ((magic & ~0xF) != MKBETAG('G', '2', 'M', '0') ||

        (magic & 0xF) < 2 || (magic & 0xF) > 4) {

        av_log(avctx, AV_LOG_ERROR, "Wrong magic %08X\n", magic);

        return AVERROR_INVALIDDATA;

    }



    if ((magic & 0xF) != 4) {

        av_log(avctx, AV_LOG_ERROR, "G2M2 and G2M3 are not yet supported\n");

        return AVERROR(ENOSYS);

    }



    while (bytestream2_get_bytes_left(&bc) > 5) {

        chunk_size = bytestream2_get_le32(&bc) - 1;

        chunk_type = bytestream2_get_byte(&bc);

        if (chunk_size > bytestream2_get_bytes_left(&bc)) {

            av_log(avctx, AV_LOG_ERROR, "Invalid chunk size %d type %02X\n",

                   chunk_size, chunk_type);

            break;

        }

        switch (chunk_type) {

        case FRAME_INFO:

            c->got_header = 0;

            if (chunk_size < 21) {

                av_log(avctx, AV_LOG_ERROR, "Invalid frame info size %d\n",

                       chunk_size);

                break;

            }

            c->width  = bytestream2_get_be32(&bc);

            c->height = bytestream2_get_be32(&bc);

            if (c->width  < 16 || c->width  > avctx->width ||

                c->height < 16 || c->height > avctx->height) {

                av_log(avctx, AV_LOG_ERROR,

                       "Invalid frame dimensions %dx%d\n",

                       c->width, c->height);

                c->width = c->height = 0;

                bytestream2_skip(&bc, bytestream2_get_bytes_left(&bc));

            }

            if (c->width != avctx->width || c->height != avctx->height)

                avcodec_set_dimensions(avctx, c->width, c->height);

            c->compression = bytestream2_get_be32(&bc);

            if (c->compression != 2 && c->compression != 3) {

                av_log(avctx, AV_LOG_ERROR,

                       "Unknown compression method %d\n",

                       c->compression);

                return AVERROR_PATCHWELCOME;

            }

            c->tile_width  = bytestream2_get_be32(&bc);

            c->tile_height = bytestream2_get_be32(&bc);

            if (!c->tile_width || !c->tile_height) {

                av_log(avctx, AV_LOG_ERROR,

                       "Invalid tile dimensions %dx%d\n",

                       c->tile_width, c->tile_height);

                return AVERROR_INVALIDDATA;

            }

            c->tiles_x = (c->width  + c->tile_width  - 1) / c->tile_width;

            c->tiles_y = (c->height + c->tile_height - 1) / c->tile_height;

            c->bpp = bytestream2_get_byte(&bc);

            chunk_size -= 21;

            bytestream2_skip(&bc, chunk_size);

            if (g2m_init_buffers(c))

                return AVERROR(ENOMEM);

            got_header = 1;

            break;

        case TILE_DATA:

            if (!c->tiles_x || !c->tiles_y) {

                av_log(avctx, AV_LOG_WARNING,

                       "No frame header - skipping tile\n");

                bytestream2_skip(&bc, bytestream2_get_bytes_left(&bc));

                break;

            }

            if (chunk_size < 2) {

                av_log(avctx, AV_LOG_ERROR, "Invalid tile data size %d\n",

                       chunk_size);

                break;

            }

            c->tile_x = bytestream2_get_byte(&bc);

            c->tile_y = bytestream2_get_byte(&bc);

            if (c->tile_x >= c->tiles_x || c->tile_y >= c->tiles_y) {

                av_log(avctx, AV_LOG_ERROR,

                       "Invalid tile pos %d,%d (in %dx%d grid)\n",

                       c->tile_x, c->tile_y, c->tiles_x, c->tiles_y);

                break;

            }

            chunk_size -= 2;

            ret = 0;

            switch (c->compression) {

            case COMPR_EPIC_J_B:

                av_log(avctx, AV_LOG_ERROR,

                       "ePIC j-b compression is not implemented yet\n");

                return AVERROR(ENOSYS);

            case COMPR_KEMPF_J_B:

                ret = kempf_decode_tile(c, c->tile_x, c->tile_y,

                                        buf + bytestream2_tell(&bc),

                                        chunk_size);

                break;

            }

            if (ret && c->framebuf)

                av_log(avctx, AV_LOG_ERROR, "Error decoding tile %d,%d\n",

                       c->tile_x, c->tile_y);

            bytestream2_skip(&bc, chunk_size);

            break;

        case CURSOR_POS:

            if (chunk_size < 5) {

                av_log(avctx, AV_LOG_ERROR, "Invalid cursor pos size %d\n",

                       chunk_size);

                break;

            }

            c->cursor_x = bytestream2_get_be16(&bc);

            c->cursor_y = bytestream2_get_be16(&bc);

            bytestream2_skip(&bc, chunk_size - 4);

            break;

        case CURSOR_SHAPE:

            if (chunk_size < 8) {

                av_log(avctx, AV_LOG_ERROR, "Invalid cursor data size %d\n",

                       chunk_size);

                break;

            }

            bytestream2_init(&tbc, buf + bytestream2_tell(&bc),

                             chunk_size - 4);

            cur_size        = bytestream2_get_be32(&tbc);

            c->cursor_w     = bytestream2_get_byte(&tbc);

            c->cursor_h     = bytestream2_get_byte(&tbc);

            c->cursor_hot_x = bytestream2_get_byte(&tbc);

            c->cursor_hot_y = bytestream2_get_byte(&tbc);

            c->cursor_fmt   = bytestream2_get_byte(&tbc);

            if (cur_size >= chunk_size ||

                c->cursor_w * c->cursor_h / 4 > cur_size) {

                av_log(avctx, AV_LOG_ERROR, "Invalid cursor data size %d\n",

                       chunk_size);

                break;

            }

            g2m_load_cursor(c, &tbc);

            bytestream2_skip(&bc, chunk_size);

            break;

        case CHUNK_CC:

        case CHUNK_CD:

            bytestream2_skip(&bc, chunk_size);

            break;

        default:

            av_log(avctx, AV_LOG_WARNING, "Skipping chunk type %02X\n",

                   chunk_type);

            bytestream2_skip(&bc, chunk_size);

        }

    }

    if (got_header)

        c->got_header = 1;



    if (c->width && c->height && c->framebuf) {

        if ((ret = ff_get_buffer(avctx, pic, 0)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return ret;

        }



        pic->key_frame = got_header;

        pic->pict_type = got_header ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;



        for (i = 0; i < avctx->height; i++)

            memcpy(pic->data[0] + i * pic->linesize[0],

                   c->framebuf  + i * c->framebuf_stride,

                   c->width * 3);

        g2m_paint_cursor(c, pic->data[0], pic->linesize[0]);



        *got_picture_ptr = 1;

    }



    return buf_size;

}
