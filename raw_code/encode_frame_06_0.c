static int encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                        const AVFrame *pic, int *got_packet)

{

    ProresContext *ctx = avctx->priv_data;

    uint8_t *orig_buf, *buf, *slice_hdr, *slice_sizes, *tmp;

    uint8_t *picture_size_pos;

    PutBitContext pb;

    int x, y, i, mb, q = 0;

    int sizes[4] = { 0 };

    int slice_hdr_size = 2 + 2 * (ctx->num_planes - 1);

    int frame_size, picture_size, slice_size;

    int mbs_per_slice = ctx->mbs_per_slice;

    int pkt_size, ret;



    *avctx->coded_frame           = *pic;

    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;



    pkt_size = ctx->mb_width * ctx->mb_height * 64 * 3 * 12

               + ctx->num_slices * 2 + 200 + FF_MIN_BUFFER_SIZE;



    if ((ret = ff_alloc_packet(pkt, pkt_size)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet.\n");

        return ret;

    }



    orig_buf = pkt->data;



    // frame atom

    orig_buf += 4;                              // frame size

    bytestream_put_be32  (&orig_buf, FRAME_ID); // frame container ID

    buf = orig_buf;



    // frame header

    tmp = buf;

    buf += 2;                                   // frame header size will be stored here

    bytestream_put_be16  (&buf, 0);             // version 1

    bytestream_put_buffer(&buf, "Lavc", 4);     // creator

    bytestream_put_be16  (&buf, avctx->width);

    bytestream_put_be16  (&buf, avctx->height);

    bytestream_put_byte  (&buf, ctx->chroma_factor << 6); // frame flags

    bytestream_put_byte  (&buf, 0);             // reserved

    bytestream_put_byte  (&buf, 0);             // primaries

    bytestream_put_byte  (&buf, 0);             // transfer function

    bytestream_put_byte  (&buf, 6);             // colour matrix - ITU-R BT.601-4

    bytestream_put_byte  (&buf, 0x40);          // source format and alpha information

    bytestream_put_byte  (&buf, 0);             // reserved

    bytestream_put_byte  (&buf, 0x03);          // matrix flags - both matrices are present

    // luma quantisation matrix

    for (i = 0; i < 64; i++)

        bytestream_put_byte(&buf, ctx->profile_info->quant[i]);

    // chroma quantisation matrix

    for (i = 0; i < 64; i++)

        bytestream_put_byte(&buf, ctx->profile_info->quant[i]);

    bytestream_put_be16  (&tmp, buf - orig_buf); // write back frame header size



    // picture header

    picture_size_pos = buf + 1;

    bytestream_put_byte  (&buf, 0x40);          // picture header size (in bits)

    buf += 4;                                   // picture data size will be stored here

    bytestream_put_be16  (&buf, ctx->num_slices); // total number of slices

    bytestream_put_byte  (&buf, av_log2(ctx->mbs_per_slice) << 4); // slice width and height in MBs



    // seek table - will be filled during slice encoding

    slice_sizes = buf;

    buf += ctx->num_slices * 2;



    // slices

    for (y = 0; y < ctx->mb_height; y++) {

        mbs_per_slice = ctx->mbs_per_slice;

        for (x = mb = 0; x < ctx->mb_width; x += mbs_per_slice, mb++) {

            while (ctx->mb_width - x < mbs_per_slice)

                mbs_per_slice >>= 1;

            q = find_slice_quant(avctx, pic, (mb + 1) * TRELLIS_WIDTH, x, y,

                                 mbs_per_slice);

        }



        for (x = ctx->slices_width - 1; x >= 0; x--) {

            ctx->slice_q[x] = ctx->nodes[q].quant;

            q = ctx->nodes[q].prev_node;

        }



        mbs_per_slice = ctx->mbs_per_slice;

        for (x = mb = 0; x < ctx->mb_width; x += mbs_per_slice, mb++) {

            q = ctx->slice_q[mb];



            while (ctx->mb_width - x < mbs_per_slice)

                mbs_per_slice >>= 1;



            bytestream_put_byte(&buf, slice_hdr_size << 3);

            slice_hdr = buf;

            buf += slice_hdr_size - 1;

            init_put_bits(&pb, buf, (pkt_size - (buf - orig_buf)) * 8);

            encode_slice(avctx, pic, &pb, sizes, x, y, q, mbs_per_slice);



            bytestream_put_byte(&slice_hdr, q);

            slice_size = slice_hdr_size + sizes[ctx->num_planes - 1];

            for (i = 0; i < ctx->num_planes - 1; i++) {

                bytestream_put_be16(&slice_hdr, sizes[i]);

                slice_size += sizes[i];

            }

            bytestream_put_be16(&slice_sizes, slice_size);

            buf += slice_size - slice_hdr_size;

        }

    }



    orig_buf -= 8;

    frame_size = buf - orig_buf;

    picture_size = buf - picture_size_pos - 6;

    bytestream_put_be32(&orig_buf, frame_size);

    bytestream_put_be32(&picture_size_pos, picture_size);



    pkt->size   = frame_size;

    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}
