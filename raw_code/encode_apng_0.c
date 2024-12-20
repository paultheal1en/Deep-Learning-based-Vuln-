static int encode_apng(AVCodecContext *avctx, AVPacket *pkt,

                       const AVFrame *pict, int *got_packet)

{

    PNGEncContext *s = avctx->priv_data;

    int ret;

    int enc_row_size;

    size_t max_packet_size;

    APNGFctlChunk fctl_chunk = {0};



    if (pict && avctx->codec_id == AV_CODEC_ID_APNG && s->color_type == PNG_COLOR_TYPE_PALETTE) {

        uint32_t checksum = ~av_crc(av_crc_get_table(AV_CRC_32_IEEE_LE), ~0U, pict->data[1], 256 * sizeof(uint32_t));



        if (avctx->frame_number == 0) {

            s->palette_checksum = checksum;

        } else if (checksum != s->palette_checksum) {

            av_log(avctx, AV_LOG_ERROR,

                   "Input contains more than one unique palette. APNG does not support multiple palettes.\n");

            return -1;

        }

    }



    enc_row_size    = deflateBound(&s->zstream, (avctx->width * s->bits_per_pixel + 7) >> 3);

    max_packet_size =

        AV_INPUT_BUFFER_MIN_SIZE + // headers

        avctx->height * (

            enc_row_size +

            (4 + 12) * (((int64_t)enc_row_size + IOBUF_SIZE - 1) / IOBUF_SIZE) // fdAT * ceil(enc_row_size / IOBUF_SIZE)

        );

    if (max_packet_size > INT_MAX)

        return AVERROR(ENOMEM);



    if (avctx->frame_number == 0) {

        if (!pict)

            return AVERROR(EINVAL);



        s->bytestream = avctx->extradata = av_malloc(FF_MIN_BUFFER_SIZE);

        if (!avctx->extradata)

            return AVERROR(ENOMEM);



        ret = encode_headers(avctx, pict);

        if (ret < 0)

            return ret;



        avctx->extradata_size = s->bytestream - avctx->extradata;



        s->last_frame_packet = av_malloc(max_packet_size);

        if (!s->last_frame_packet)

            return AVERROR(ENOMEM);

    } else if (s->last_frame) {

        ret = ff_alloc_packet2(avctx, pkt, max_packet_size, 0);

        if (ret < 0)

            return ret;



        memcpy(pkt->data, s->last_frame_packet, s->last_frame_packet_size);

        pkt->size = s->last_frame_packet_size;

        pkt->pts = pkt->dts = s->last_frame->pts;

    }



    if (pict) {

        s->bytestream_start =

        s->bytestream       = s->last_frame_packet;

        s->bytestream_end   = s->bytestream + max_packet_size;



        // We're encoding the frame first, so we have to do a bit of shuffling around

        // to have the image data write to the correct place in the buffer

        fctl_chunk.sequence_number = s->sequence_number;

        ++s->sequence_number;

        s->bytestream += 26 + 12;



        ret = apng_encode_frame(avctx, pict, &fctl_chunk, &s->last_frame_fctl);

        if (ret < 0)

            return ret;



        fctl_chunk.delay_num = 0; // delay filled in during muxing

        fctl_chunk.delay_den = 0;

    } else {

        s->last_frame_fctl.dispose_op = APNG_DISPOSE_OP_NONE;

    }



    if (s->last_frame) {

        uint8_t* last_fctl_chunk_start = pkt->data;

        uint8_t buf[26];



        AV_WB32(buf + 0, s->last_frame_fctl.sequence_number);

        AV_WB32(buf + 4, s->last_frame_fctl.width);

        AV_WB32(buf + 8, s->last_frame_fctl.height);

        AV_WB32(buf + 12, s->last_frame_fctl.x_offset);

        AV_WB32(buf + 16, s->last_frame_fctl.y_offset);

        AV_WB16(buf + 20, s->last_frame_fctl.delay_num);

        AV_WB16(buf + 22, s->last_frame_fctl.delay_den);

        buf[24] = s->last_frame_fctl.dispose_op;

        buf[25] = s->last_frame_fctl.blend_op;

        png_write_chunk(&last_fctl_chunk_start, MKTAG('f', 'c', 'T', 'L'), buf, 26);



        *got_packet = 1;

    }



    if (pict) {

        if (!s->last_frame) {

            s->last_frame = av_frame_alloc();

            if (!s->last_frame)

                return AVERROR(ENOMEM);

        } else if (s->last_frame_fctl.dispose_op != APNG_DISPOSE_OP_PREVIOUS) {

            if (!s->prev_frame) {

                s->prev_frame = av_frame_alloc();

                if (!s->prev_frame)

                    return AVERROR(ENOMEM);



                s->prev_frame->format = pict->format;

                s->prev_frame->width = pict->width;

                s->prev_frame->height = pict->height;

                if ((ret = av_frame_get_buffer(s->prev_frame, 32)) < 0)

                    return ret;

            }



            // Do disposal, but not blending

            memcpy(s->prev_frame->data[0], s->last_frame->data[0],

                   s->last_frame->linesize[0] * s->last_frame->height);

            if (s->last_frame_fctl.dispose_op == APNG_DISPOSE_OP_BACKGROUND) {

                uint32_t y;

                uint8_t bpp = (s->bits_per_pixel + 7) >> 3;

                for (y = s->last_frame_fctl.y_offset; y < s->last_frame_fctl.y_offset + s->last_frame_fctl.height; ++y) {

                    size_t row_start = s->last_frame->linesize[0] * y + bpp * s->last_frame_fctl.x_offset;

                    memset(s->prev_frame->data[0] + row_start, 0, bpp * s->last_frame_fctl.width);

                }

            }

        }



        av_frame_unref(s->last_frame);

        ret = av_frame_ref(s->last_frame, (AVFrame*)pict);

        if (ret < 0)

            return ret;



        s->last_frame_fctl = fctl_chunk;

        s->last_frame_packet_size = s->bytestream - s->bytestream_start;

    } else {

        av_frame_free(&s->last_frame);

    }



    return 0;

}
