static int encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                        const AVFrame *frame, int *got_packet)

{

    SgiContext *s = avctx->priv_data;

    const AVFrame * const p = frame;

    PutByteContext pbc;

    uint8_t *in_buf, *encode_buf;

    int x, y, z, length, tablesize, ret;

    unsigned int width, height, depth, dimension;

    unsigned int bytes_per_channel, pixmax, put_be;



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



#if FF_API_CODER_TYPE

FF_DISABLE_DEPRECATION_WARNINGS

    if (avctx->coder_type == FF_CODER_TYPE_RAW)

        s->rle = 0;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    width  = avctx->width;

    height = avctx->height;

    bytes_per_channel = 1;

    pixmax = 0xFF;

    put_be = HAVE_BIGENDIAN;



    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_GRAY8:

        dimension = SGI_SINGLE_CHAN;

        depth     = SGI_GRAYSCALE;

        break;

    case AV_PIX_FMT_RGB24:

        dimension = SGI_MULTI_CHAN;

        depth     = SGI_RGB;

        break;

    case AV_PIX_FMT_RGBA:

        dimension = SGI_MULTI_CHAN;

        depth     = SGI_RGBA;

        break;

    case AV_PIX_FMT_GRAY16LE:

        put_be = !HAVE_BIGENDIAN;

    case AV_PIX_FMT_GRAY16BE:

        bytes_per_channel = 2;

        pixmax = 0xFFFF;

        dimension = SGI_SINGLE_CHAN;

        depth     = SGI_GRAYSCALE;

        break;

    case AV_PIX_FMT_RGB48LE:

        put_be = !HAVE_BIGENDIAN;

    case AV_PIX_FMT_RGB48BE:

        bytes_per_channel = 2;

        pixmax = 0xFFFF;

        dimension = SGI_MULTI_CHAN;

        depth     = SGI_RGB;

        break;

    case AV_PIX_FMT_RGBA64LE:

        put_be = !HAVE_BIGENDIAN;

    case AV_PIX_FMT_RGBA64BE:

        bytes_per_channel = 2;

        pixmax = 0xFFFF;

        dimension = SGI_MULTI_CHAN;

        depth     = SGI_RGBA;

        break;

    default:

        return AVERROR_INVALIDDATA;

    }



    tablesize = depth * height * 4;

    length = SGI_HEADER_SIZE;

    if (!s->rle)

        length += depth * height * width;

    else // assume sgi_rle_encode() produces at most 2x size of input

        length += tablesize * 2 + depth * height * (2 * width + 1);



    if ((ret = ff_alloc_packet(pkt, bytes_per_channel * length)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet of size %d.\n", length);

        return ret;

    }



    bytestream2_init_writer(&pbc, pkt->data, pkt->size);



    /* Encode header. */

    bytestream2_put_be16(&pbc, SGI_MAGIC);

    bytestream2_put_byte(&pbc, s->rle); /* RLE 1 - VERBATIM 0 */

    bytestream2_put_byte(&pbc, bytes_per_channel);

    bytestream2_put_be16(&pbc, dimension);

    bytestream2_put_be16(&pbc, width);

    bytestream2_put_be16(&pbc, height);

    bytestream2_put_be16(&pbc, depth);



    bytestream2_put_be32(&pbc, 0L); /* pixmin */

    bytestream2_put_be32(&pbc, pixmax);

    bytestream2_put_be32(&pbc, 0L); /* dummy */



    /* name */

    bytestream2_skip_p(&pbc, 80);



    /* colormap */

    bytestream2_put_be32(&pbc, 0L);



    /* The rest of the 512 byte header is unused. */

    bytestream2_skip_p(&pbc, 404);



    if (s->rle) {

        PutByteContext taboff_pcb, tablen_pcb;



        /* Skip RLE offset table. */

        bytestream2_init_writer(&taboff_pcb, pbc.buffer, tablesize);

        bytestream2_skip_p(&pbc, tablesize);



        /* Skip RLE length table. */

        bytestream2_init_writer(&tablen_pcb, pbc.buffer, tablesize);

        bytestream2_skip_p(&pbc, tablesize);



        /* Make an intermediate consecutive buffer. */

        if (!(encode_buf = av_malloc(width * bytes_per_channel)))

            return AVERROR(ENOMEM);



        for (z = 0; z < depth; z++) {

            in_buf = p->data[0] + p->linesize[0] * (height - 1) + z * bytes_per_channel;



            for (y = 0; y < height; y++) {

                bytestream2_put_be32(&taboff_pcb, bytestream2_tell_p(&pbc));



                for (x = 0; x < width * bytes_per_channel; x += bytes_per_channel)

                    encode_buf[x] = in_buf[depth * x];



                length = sgi_rle_encode(&pbc, encode_buf, width,

                                        bytes_per_channel);

                if (length < 1) {

                    av_free(encode_buf);

                    return AVERROR_INVALIDDATA;

                }



                bytestream2_put_be32(&tablen_pcb, length);

                in_buf -= p->linesize[0];

            }

        }



        av_free(encode_buf);

    } else {

        for (z = 0; z < depth; z++) {

            in_buf = p->data[0] + p->linesize[0] * (height - 1) + z * bytes_per_channel;



            for (y = 0; y < height; y++) {

                for (x = 0; x < width * depth; x += depth)

                    if (bytes_per_channel == 1)

                        bytestream2_put_byte(&pbc, in_buf[x]);

                    else

                        if (put_be)

                            bytestream2_put_be16(&pbc, ((uint16_t *)in_buf)[x]);

                        else

                            bytestream2_put_le16(&pbc, ((uint16_t *)in_buf)[x]);



                in_buf -= p->linesize[0];

            }

        }

    }



    /* total length */

    pkt->size   = bytestream2_tell_p(&pbc);

    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}
