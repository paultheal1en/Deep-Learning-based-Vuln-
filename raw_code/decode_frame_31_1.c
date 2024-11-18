static int decode_frame(AVCodecContext *avctx, void *data, int *data_size, AVPacket *pkt)

{

    BinkContext * const c = avctx->priv_data;

    GetBitContext gb;

    int blk;

    int i, j, plane, plane_idx, bx, by;

    uint8_t *dst, *prev, *ref, *ref_start, *ref_end;

    int v, col[2];

    const uint8_t *scan;

    int xoff, yoff;

    DECLARE_ALIGNED_16(DCTELEM, block[64]);

    DECLARE_ALIGNED_16(uint8_t, ublock[64]);

    int coordmap[64];



    if(c->pic.data[0])

        avctx->release_buffer(avctx, &c->pic);



    if(avctx->get_buffer(avctx, &c->pic) < 0){

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    init_get_bits(&gb, pkt->data, pkt->size*8);

    if (c->version >= 'i')

        skip_bits_long(&gb, 32);



    for (plane = 0; plane < 3; plane++) {

        const int stride = c->pic.linesize[plane];

        int bw = plane ? (avctx->width  + 15) >> 4 : (avctx->width  + 7) >> 3;

        int bh = plane ? (avctx->height + 15) >> 4 : (avctx->height + 7) >> 3;

        int width = avctx->width >> !!plane;



        init_lengths(c, FFMAX(width, 8), bw);

        for (i = 0; i < BINK_NB_SRC; i++)

            read_bundle(&gb, c, i);



        plane_idx = (!plane || !c->swap_planes) ? plane : (plane ^ 3);

        ref_start = c->last.data[plane_idx];

        ref_end   = c->last.data[plane_idx]

                    + (bw - 1 + c->last.linesize[plane_idx] * (bh - 1)) * 8;



        for (i = 0; i < 64; i++)

            coordmap[i] = (i & 7) + (i >> 3) * stride;



        for (by = 0; by < bh; by++) {

            if (read_block_types(avctx, &gb, &c->bundle[BINK_SRC_BLOCK_TYPES]) < 0)

                return -1;

            if (read_block_types(avctx, &gb, &c->bundle[BINK_SRC_SUB_BLOCK_TYPES]) < 0)

                return -1;

            if (read_colors(&gb, &c->bundle[BINK_SRC_COLORS], c) < 0)

                return -1;

            if (read_patterns(avctx, &gb, &c->bundle[BINK_SRC_PATTERN]) < 0)

                return -1;

            if (read_motion_values(avctx, &gb, &c->bundle[BINK_SRC_X_OFF]) < 0)

                return -1;

            if (read_motion_values(avctx, &gb, &c->bundle[BINK_SRC_Y_OFF]) < 0)

                return -1;

            if (read_dcs(avctx, &gb, &c->bundle[BINK_SRC_INTRA_DC], DC_START_BITS, 0) < 0)

                return -1;

            if (read_dcs(avctx, &gb, &c->bundle[BINK_SRC_INTER_DC], DC_START_BITS, 1) < 0)

                return -1;

            if (read_runs(avctx, &gb, &c->bundle[BINK_SRC_RUN]) < 0)

                return -1;



            if (by == bh)

                break;

            dst  = c->pic.data[plane_idx]  + 8*by*stride;

            prev = c->last.data[plane_idx] + 8*by*stride;

            for (bx = 0; bx < bw; bx++, dst += 8, prev += 8) {

                blk = get_value(c, BINK_SRC_BLOCK_TYPES);

                // 16x16 block type on odd line means part of the already decoded block, so skip it

                if ((by & 1) && blk == SCALED_BLOCK) {

                    bx++;

                    dst  += 8;

                    prev += 8;

                    continue;

                }

                switch (blk) {

                case SKIP_BLOCK:

                    c->dsp.put_pixels_tab[1][0](dst, prev, stride, 8);

                    break;

                case SCALED_BLOCK:

                    blk = get_value(c, BINK_SRC_SUB_BLOCK_TYPES);

                    switch (blk) {

                    case RUN_BLOCK:

                        scan = bink_patterns[get_bits(&gb, 4)];

                        i = 0;

                        do {

                            int run = get_value(c, BINK_SRC_RUN) + 1;



                            i += run;

                            if (i > 64) {

                                av_log(avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                                return -1;

                            }

                            if (get_bits1(&gb)) {

                                v = get_value(c, BINK_SRC_COLORS);

                                for (j = 0; j < run; j++)

                                    ublock[*scan++] = v;

                            } else {

                                for (j = 0; j < run; j++)

                                    ublock[*scan++] = get_value(c, BINK_SRC_COLORS);

                            }

                        } while (i < 63);

                        if (i == 63)

                            ublock[*scan++] = get_value(c, BINK_SRC_COLORS);

                        break;

                    case INTRA_BLOCK:

                        c->dsp.clear_block(block);

                        block[0] = get_value(c, BINK_SRC_INTRA_DC);

                        read_dct_coeffs(&gb, block, c->scantable.permutated, 1);

                        c->dsp.idct(block);

                        c->dsp.put_pixels_nonclamped(block, ublock, 8);

                        break;

                    case FILL_BLOCK:

                        v = get_value(c, BINK_SRC_COLORS);

                        c->dsp.fill_block_tab[0](dst, v, stride, 16);

                        break;

                    case PATTERN_BLOCK:

                        for (i = 0; i < 2; i++)

                            col[i] = get_value(c, BINK_SRC_COLORS);

                        for (j = 0; j < 8; j++) {

                            v = get_value(c, BINK_SRC_PATTERN);

                            for (i = 0; i < 8; i++, v >>= 1)

                                ublock[i + j*8] = col[v & 1];

                        }

                        break;

                    case RAW_BLOCK:

                        for (j = 0; j < 8; j++)

                            for (i = 0; i < 8; i++)

                                ublock[i + j*8] = get_value(c, BINK_SRC_COLORS);

                        break;

                    default:

                        av_log(avctx, AV_LOG_ERROR, "Incorrect 16x16 block type %d\n", blk);

                        return -1;

                    }

                    if (blk != FILL_BLOCK)

                        c->dsp.scale_block(ublock, dst, stride);

                    bx++;

                    dst  += 8;

                    prev += 8;

                    break;

                case MOTION_BLOCK:

                    xoff = get_value(c, BINK_SRC_X_OFF);

                    yoff = get_value(c, BINK_SRC_Y_OFF);

                    ref = prev + xoff + yoff * stride;

                    if (ref < ref_start || ref > ref_end) {

                        av_log(avctx, AV_LOG_ERROR, "Copy out of bounds @%d, %d\n",

                               bx*8 + xoff, by*8 + yoff);

                        return -1;

                    }

                    c->dsp.put_pixels_tab[1][0](dst, ref, stride, 8);

                    break;

                case RUN_BLOCK:

                    scan = bink_patterns[get_bits(&gb, 4)];

                    i = 0;

                    do {

                        int run = get_value(c, BINK_SRC_RUN) + 1;



                        i += run;

                        if (i > 64) {

                            av_log(avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                            return -1;

                        }

                        if (get_bits1(&gb)) {

                            v = get_value(c, BINK_SRC_COLORS);

                            for (j = 0; j < run; j++)

                                dst[coordmap[*scan++]] = v;

                        } else {

                            for (j = 0; j < run; j++)

                                dst[coordmap[*scan++]] = get_value(c, BINK_SRC_COLORS);

                        }

                    } while (i < 63);

                    if (i == 63)

                        dst[coordmap[*scan++]] = get_value(c, BINK_SRC_COLORS);

                    break;

                case RESIDUE_BLOCK:

                    xoff = get_value(c, BINK_SRC_X_OFF);

                    yoff = get_value(c, BINK_SRC_Y_OFF);

                    ref = prev + xoff + yoff * stride;

                    if (ref < ref_start || ref > ref_end) {

                        av_log(avctx, AV_LOG_ERROR, "Copy out of bounds @%d, %d\n",

                               bx*8 + xoff, by*8 + yoff);

                        return -1;

                    }

                    c->dsp.put_pixels_tab[1][0](dst, ref, stride, 8);

                    c->dsp.clear_block(block);

                    v = get_bits(&gb, 7);

                    read_residue(&gb, block, v);

                    c->dsp.add_pixels8(dst, block, stride);

                    break;

                case INTRA_BLOCK:

                    c->dsp.clear_block(block);

                    block[0] = get_value(c, BINK_SRC_INTRA_DC);

                    read_dct_coeffs(&gb, block, c->scantable.permutated, 1);

                    c->dsp.idct_put(dst, stride, block);

                    break;

                case FILL_BLOCK:

                    v = get_value(c, BINK_SRC_COLORS);

                    c->dsp.fill_block_tab[1](dst, v, stride, 8);

                    break;

                case INTER_BLOCK:

                    xoff = get_value(c, BINK_SRC_X_OFF);

                    yoff = get_value(c, BINK_SRC_Y_OFF);

                    ref = prev + xoff + yoff * stride;

                    c->dsp.put_pixels_tab[1][0](dst, ref, stride, 8);

                    c->dsp.clear_block(block);

                    block[0] = get_value(c, BINK_SRC_INTER_DC);

                    read_dct_coeffs(&gb, block, c->scantable.permutated, 0);

                    c->dsp.idct_add(dst, stride, block);

                    break;

                case PATTERN_BLOCK:

                    for (i = 0; i < 2; i++)

                        col[i] = get_value(c, BINK_SRC_COLORS);

                    for (i = 0; i < 8; i++) {

                        v = get_value(c, BINK_SRC_PATTERN);

                        for (j = 0; j < 8; j++, v >>= 1)

                            dst[i*stride + j] = col[v & 1];

                    }

                    break;

                case RAW_BLOCK:

                    for (i = 0; i < 8; i++)

                        memcpy(dst + i*stride, c->bundle[BINK_SRC_COLORS].cur_ptr + i*8, 8);

                    c->bundle[BINK_SRC_COLORS].cur_ptr += 64;

                    break;

                default:

                    av_log(avctx, AV_LOG_ERROR, "Unknown block type %d\n", blk);

                    return -1;

                }

            }

        }

        if (get_bits_count(&gb) & 0x1F) //next plane data starts at 32-bit boundary

            skip_bits_long(&gb, 32 - (get_bits_count(&gb) & 0x1F));

    }

    emms_c();



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = c->pic;



    FFSWAP(AVFrame, c->pic, c->last);



    /* always report that the buffer was completely consumed */

    return pkt->size;

}
