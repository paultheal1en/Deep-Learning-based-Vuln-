static int encode_plane(AVCodecContext *avctx, uint8_t *src,

                        uint8_t *dst, int stride,

                        int width, int height, PutByteContext *pb)

{

    UtvideoContext *c        = avctx->priv_data;

    uint8_t  lengths[256];

    uint64_t counts[256]     = { 0 };



    HuffEntry he[256];



    uint32_t offset = 0, slice_len = 0;

    int      i, sstart, send = 0;

    int      symbol;



    /* Do prediction / make planes */

    switch (c->frame_pred) {

    case PRED_NONE:

        for (i = 0; i < c->slices; i++) {

            sstart = send;

            send   = height * (i + 1) / c->slices;

            write_plane(src + sstart * stride, dst + sstart * width,

                        stride, width, send - sstart);

        }

        break;

    case PRED_LEFT:

        for (i = 0; i < c->slices; i++) {

            sstart = send;

            send   = height * (i + 1) / c->slices;

            left_predict(src + sstart * stride, dst + sstart * width,

                         stride, width, send - sstart);

        }

        break;

    case PRED_MEDIAN:

        for (i = 0; i < c->slices; i++) {

            sstart = send;

            send   = height * (i + 1) / c->slices;

            median_predict(c, src + sstart * stride, dst + sstart * width,

                           stride, width, send - sstart);

        }

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown prediction mode: %d\n",

               c->frame_pred);

        return AVERROR_OPTION_NOT_FOUND;

    }



    /* Count the usage of values */

    count_usage(dst, width, height, counts);



    /* Check for a special case where only one symbol was used */

    for (symbol = 0; symbol < 256; symbol++) {

        /* If non-zero count is found, see if it matches width * height */

        if (counts[symbol]) {

            /* Special case if only one symbol was used */

            if (counts[symbol] == width * height) {

                /*

                 * Write a zero for the single symbol

                 * used in the plane, else 0xFF.

                 */

                for (i = 0; i < 256; i++) {

                    if (i == symbol)

                        bytestream2_put_byte(pb, 0);

                    else

                        bytestream2_put_byte(pb, 0xFF);

                }



                /* Write zeroes for lengths */

                for (i = 0; i < c->slices; i++)

                    bytestream2_put_le32(pb, 0);



                /* And that's all for that plane folks */

                return 0;

            }

            break;

        }

    }



    /* Calculate huffman lengths */

    ff_huff_gen_len_table(lengths, counts);



    /*

     * Write the plane's header into the output packet:

     * - huffman code lengths (256 bytes)

     * - slice end offsets (gotten from the slice lengths)

     */

    for (i = 0; i < 256; i++) {

        bytestream2_put_byte(pb, lengths[i]);



        he[i].len = lengths[i];

        he[i].sym = i;

    }



    /* Calculate the huffman codes themselves */

    calculate_codes(he);



    send = 0;

    for (i = 0; i < c->slices; i++) {

        sstart  = send;

        send    = height * (i + 1) / c->slices;



        /*

         * Write the huffman codes to a buffer,

         * get the offset in bits and convert to bytes.

         */

        offset += write_huff_codes(dst + sstart * width, c->slice_bits,

                                   width * (send - sstart), width,

                                   send - sstart, he) >> 3;



        slice_len = offset - slice_len;



        /* Byteswap the written huffman codes */

        c->dsp.bswap_buf((uint32_t *) c->slice_bits,

                         (uint32_t *) c->slice_bits,

                         slice_len >> 2);



        /* Write the offset to the stream */

        bytestream2_put_le32(pb, offset);



        /* Seek to the data part of the packet */

        bytestream2_seek_p(pb, 4 * (c->slices - i - 1) +

                           offset - slice_len, SEEK_CUR);



        /* Write the slices' data into the output packet */

        bytestream2_put_buffer(pb, c->slice_bits, slice_len);



        /* Seek back to the slice offsets */

        bytestream2_seek_p(pb, -4 * (c->slices - i - 1) - offset,

                           SEEK_CUR);



        slice_len = offset;

    }



    /* And at the end seek to the end of written slice(s) */

    bytestream2_seek_p(pb, offset, SEEK_CUR);



    return 0;

}
