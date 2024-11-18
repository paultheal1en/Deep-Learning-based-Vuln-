static int bink_decode_plane(BinkContext *c, AVFrame *frame, BitstreamContext *bc,

                             int plane_idx, int is_chroma)

{

    int blk, ret;

    int i, j, bx, by;

    uint8_t *dst, *prev, *ref_start, *ref_end;

    int v, col[2];

    const uint8_t *scan;

    LOCAL_ALIGNED_16(int16_t, block, [64]);

    LOCAL_ALIGNED_16(uint8_t, ublock, [64]);

    LOCAL_ALIGNED_16(int32_t, dctblock, [64]);

    int coordmap[64];



    const int stride = frame->linesize[plane_idx];

    int bw = is_chroma ? (c->avctx->width  + 15) >> 4 : (c->avctx->width  + 7) >> 3;

    int bh = is_chroma ? (c->avctx->height + 15) >> 4 : (c->avctx->height + 7) >> 3;

    int width = c->avctx->width >> is_chroma;



    init_lengths(c, FFMAX(width, 8), bw);

    for (i = 0; i < BINK_NB_SRC; i++)

        read_bundle(bc, c, i);



    ref_start = c->last->data[plane_idx] ? c->last->data[plane_idx]

                                         : frame->data[plane_idx];

    ref_end   = ref_start

                + (bw - 1 + c->last->linesize[plane_idx] * (bh - 1)) * 8;



    for (i = 0; i < 64; i++)

        coordmap[i] = (i & 7) + (i >> 3) * stride;



    for (by = 0; by < bh; by++) {

        if ((ret = read_block_types(c->avctx, bc, &c->bundle[BINK_SRC_BLOCK_TYPES])) < 0)

            return ret;

        if ((ret = read_block_types(c->avctx, bc, &c->bundle[BINK_SRC_SUB_BLOCK_TYPES])) < 0)

            return ret;

        if ((ret = read_colors(bc, &c->bundle[BINK_SRC_COLORS], c)) < 0)

            return ret;

        if ((ret = read_patterns(c->avctx, bc, &c->bundle[BINK_SRC_PATTERN])) < 0)

            return ret;

        if ((ret = read_motion_values(c->avctx, bc, &c->bundle[BINK_SRC_X_OFF])) < 0)

            return ret;

        if ((ret = read_motion_values(c->avctx, bc, &c->bundle[BINK_SRC_Y_OFF])) < 0)

            return ret;

        if ((ret = read_dcs(c->avctx, bc, &c->bundle[BINK_SRC_INTRA_DC], DC_START_BITS, 0)) < 0)

            return ret;

        if ((ret = read_dcs(c->avctx, bc, &c->bundle[BINK_SRC_INTER_DC], DC_START_BITS, 1)) < 0)

            return ret;

        if ((ret = read_runs(c->avctx, bc, &c->bundle[BINK_SRC_RUN])) < 0)

            return ret;



        if (by == bh)

            break;

        dst  = frame->data[plane_idx]  + 8*by*stride;

        prev = (c->last->data[plane_idx] ? c->last->data[plane_idx]

                                         : frame->data[plane_idx]) + 8*by*stride;

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

                c->hdsp.put_pixels_tab[1][0](dst, prev, stride, 8);

                break;

            case SCALED_BLOCK:

                blk = get_value(c, BINK_SRC_SUB_BLOCK_TYPES);

                switch (blk) {

                case RUN_BLOCK:

                    scan = bink_patterns[bitstream_read(bc, 4)];

                    i = 0;

                    do {

                        int run = get_value(c, BINK_SRC_RUN) + 1;



                        i += run;

                        if (i > 64) {

                            av_log(c->avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                            return AVERROR_INVALIDDATA;

                        }

                        if (bitstream_read_bit(bc)) {

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

                    memset(dctblock, 0, sizeof(*dctblock) * 64);

                    dctblock[0] = get_value(c, BINK_SRC_INTRA_DC);

                    read_dct_coeffs(bc, dctblock, bink_scan, bink_intra_quant, -1);

                    c->binkdsp.idct_put(ublock, 8, dctblock);

                    break;

                case FILL_BLOCK:

                    v = get_value(c, BINK_SRC_COLORS);

                    c->bdsp.fill_block_tab[0](dst, v, stride, 16);

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

                    av_log(c->avctx, AV_LOG_ERROR, "Incorrect 16x16 block type %d\n", blk);

                    return AVERROR_INVALIDDATA;

                }

                if (blk != FILL_BLOCK)

                c->binkdsp.scale_block(ublock, dst, stride);

                bx++;

                dst  += 8;

                prev += 8;

                break;

            case MOTION_BLOCK:

                ret = bink_put_pixels(c, dst, prev, stride,

                                      ref_start, ref_end);

                if (ret < 0)

                    return ret;

                break;

            case RUN_BLOCK:

                scan = bink_patterns[bitstream_read(bc, 4)];

                i = 0;

                do {

                    int run = get_value(c, BINK_SRC_RUN) + 1;



                    i += run;

                    if (i > 64) {

                        av_log(c->avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                        return AVERROR_INVALIDDATA;

                    }

                    if (bitstream_read_bit(bc)) {

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

                ret = bink_put_pixels(c, dst, prev, stride,

                                      ref_start, ref_end);

                if (ret < 0)

                    return ret;

                c->bdsp.clear_block(block);

                v = bitstream_read(bc, 7);

                read_residue(bc, block, v);

                c->binkdsp.add_pixels8(dst, block, stride);

                break;

            case INTRA_BLOCK:

                memset(dctblock, 0, sizeof(*dctblock) * 64);

                dctblock[0] = get_value(c, BINK_SRC_INTRA_DC);

                read_dct_coeffs(bc, dctblock, bink_scan, bink_intra_quant, -1);

                c->binkdsp.idct_put(dst, stride, dctblock);

                break;

            case FILL_BLOCK:

                v = get_value(c, BINK_SRC_COLORS);

                c->bdsp.fill_block_tab[1](dst, v, stride, 8);

                break;

            case INTER_BLOCK:

                ret = bink_put_pixels(c, dst, prev, stride,

                                      ref_start, ref_end);

                if (ret < 0)

                    return ret;

                memset(dctblock, 0, sizeof(*dctblock) * 64);

                dctblock[0] = get_value(c, BINK_SRC_INTER_DC);

                read_dct_coeffs(bc, dctblock, bink_scan, bink_inter_quant, -1);

                c->binkdsp.idct_add(dst, stride, dctblock);

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

                av_log(c->avctx, AV_LOG_ERROR, "Unknown block type %d\n", blk);

                return AVERROR_INVALIDDATA;

            }

        }

    }

    if (bitstream_tell(bc) & 0x1F) // next plane data starts at 32-bit boundary

        bitstream_skip(bc, 32 - (bitstream_tell(bc) & 0x1F));



    return 0;

}
