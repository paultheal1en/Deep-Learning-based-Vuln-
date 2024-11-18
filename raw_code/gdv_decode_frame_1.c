static int gdv_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame, AVPacket *avpkt)

{

    GDVContext *gdv = avctx->priv_data;

    GetByteContext *gb = &gdv->gb;

    PutByteContext *pb = &gdv->pb;

    AVFrame *frame = data;

    int ret, i, pal_size;

    const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, &pal_size);

    int compression;

    unsigned flags;

    uint8_t *dst;



    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

        return ret;

    if (pal && pal_size == AVPALETTE_SIZE)

        memcpy(gdv->pal, pal, AVPALETTE_SIZE);



    bytestream2_init(gb, avpkt->data, avpkt->size);

    bytestream2_init_writer(pb, gdv->frame, gdv->frame_size);



    flags = bytestream2_get_le32(gb);

    compression = flags & 0xF;



    rescale(gdv, gdv->frame, avctx->width, avctx->height,

            !!(flags & 0x10), !!(flags & 0x20));



    switch (compression) {

    case 1:

        memset(gdv->frame + PREAMBLE_SIZE, 0, gdv->frame_size - PREAMBLE_SIZE);

    case 0:

        if (bytestream2_get_bytes_left(gb) < 256*3)

            return AVERROR_INVALIDDATA;

        for (i = 0; i < 256; i++) {

            unsigned r = bytestream2_get_byte(gb);

            unsigned g = bytestream2_get_byte(gb);

            unsigned b = bytestream2_get_byte(gb);

            gdv->pal[i] = 0xFFU << 24 | r << 18 | g << 10 | b << 2;

        }

        break;

    case 2:

        ret = decompress_2(avctx);

        break;

    case 3:

        break;

    case 5:

        ret = decompress_5(avctx, flags >> 8);

        break;

    case 6:

        ret = decompress_68(avctx, flags >> 8, 0);

        break;

    case 8:

        ret = decompress_68(avctx, flags >> 8, 1);

        break;

    default:

        return AVERROR_INVALIDDATA;

    }



    memcpy(frame->data[1], gdv->pal, AVPALETTE_SIZE);

    dst = frame->data[0];



    if (!gdv->scale_v && !gdv->scale_h) {

        int sidx = PREAMBLE_SIZE, didx = 0;

        int y, x;



        for (y = 0; y < avctx->height; y++) {

            for (x = 0; x < avctx->width; x++) {

                dst[x+didx] = gdv->frame[x+sidx];

            }

            sidx += avctx->width;

            didx += frame->linesize[0];

        }

    } else {

        int sidx = PREAMBLE_SIZE, didx = 0;

        int y, x;



        for (y = 0; y < avctx->height; y++) {

            if (!gdv->scale_v) {

                for (x = 0; x < avctx->width; x++) {

                    dst[didx + x] = gdv->frame[sidx + x];

                }

            } else {

                for (x = 0; x < avctx->width; x++) {

                    dst[didx + x] = gdv->frame[sidx + x/2];

                }

            }

            if (!gdv->scale_h || ((y & 1) == 1)) {

                sidx += !gdv->scale_v ? avctx->width : avctx->width/2;

            }

            didx += frame->linesize[0];

        }

    }



    *got_frame = 1;



    return ret < 0 ? ret : avpkt->size;

}
