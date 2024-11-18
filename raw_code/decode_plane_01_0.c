static int decode_plane(UtvideoContext *c, int plane_no,

                        uint8_t *dst, int step, ptrdiff_t stride,

                        int width, int height,

                        const uint8_t *src, int use_pred)

{

    int i, j, slice, pix;

    int sstart, send;

    VLC vlc;

    GetBitContext gb;

    int prev, fsym;

    const int cmask = c->interlaced ? ~(1 + 2 * (!plane_no && c->avctx->pix_fmt == AV_PIX_FMT_YUV420P)) : ~(!plane_no && c->avctx->pix_fmt == AV_PIX_FMT_YUV420P);



    if (build_huff(src, &vlc, &fsym)) {

        av_log(c->avctx, AV_LOG_ERROR, "Cannot build Huffman codes\n");

        return AVERROR_INVALIDDATA;

    }

    if (fsym >= 0) { // build_huff reported a symbol to fill slices with

        send = 0;

        for (slice = 0; slice < c->slices; slice++) {

            uint8_t *dest;



            sstart = send;

            send   = (height * (slice + 1) / c->slices) & cmask;

            dest   = dst + sstart * stride;



            prev = 0x80;

            for (j = sstart; j < send; j++) {

                for (i = 0; i < width * step; i += step) {

                    pix = fsym;

                    if (use_pred) {

                        prev += pix;

                        pix   = prev;

                    }

                    dest[i] = pix;

                }

                dest += stride;

            }

        }

        return 0;

    }



    src      += 256;



    send = 0;

    for (slice = 0; slice < c->slices; slice++) {

        uint8_t *dest;

        int slice_data_start, slice_data_end, slice_size;



        sstart = send;

        send   = (height * (slice + 1) / c->slices) & cmask;

        dest   = dst + sstart * stride;



        // slice offset and size validation was done earlier

        slice_data_start = slice ? AV_RL32(src + slice * 4 - 4) : 0;

        slice_data_end   = AV_RL32(src + slice * 4);

        slice_size       = slice_data_end - slice_data_start;



        if (!slice_size) {

            av_log(c->avctx, AV_LOG_ERROR, "Plane has more than one symbol "

                   "yet a slice has a length of zero.\n");

            goto fail;

        }



        memset(c->slice_bits + slice_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

        c->bdsp.bswap_buf((uint32_t *) c->slice_bits,

                          (uint32_t *)(src + slice_data_start + c->slices * 4),

                          (slice_data_end - slice_data_start + 3) >> 2);

        init_get_bits(&gb, c->slice_bits, slice_size * 8);



        prev = 0x80;

        for (j = sstart; j < send; j++) {

            for (i = 0; i < width * step; i += step) {

                pix = get_vlc2(&gb, vlc.table, VLC_BITS, 3);

                if (pix < 0) {

                    av_log(c->avctx, AV_LOG_ERROR, "Decoding error\n");

                    goto fail;

                }

                if (use_pred) {

                    prev += pix;

                    pix   = prev;

                }

                dest[i] = pix;

            }

            if (get_bits_left(&gb) < 0) {

                av_log(c->avctx, AV_LOG_ERROR,

                        "Slice decoding ran out of bits\n");

                goto fail;

            }

            dest += stride;

        }

        if (get_bits_left(&gb) > 32)

            av_log(c->avctx, AV_LOG_WARNING,

                   "%d bits left after decoding slice\n", get_bits_left(&gb));

    }



    ff_free_vlc(&vlc);



    return 0;

fail:

    ff_free_vlc(&vlc);

    return AVERROR_INVALIDDATA;

}
