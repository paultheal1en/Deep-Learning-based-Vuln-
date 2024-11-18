static inline int mpeg2_decode_block_intra(MpegEncContext *s, int16_t *block, int n)

{

    int level, dc, diff, i, j, run;

    int component;

    RLTable *rl;

    uint8_t * const scantable = s->intra_scantable.permutated;

    const uint16_t *quant_matrix;

    const int qscale = s->qscale;

    int mismatch;



    /* DC coefficient */

    if (n < 4) {

        quant_matrix = s->intra_matrix;

        component = 0;

    } else {

        quant_matrix = s->chroma_intra_matrix;

        component = (n & 1) + 1;

    }

    diff = decode_dc(&s->gb, component);

    if (diff >= 0xffff)

        return -1;

    dc  = s->last_dc[component];

    dc += diff;

    s->last_dc[component] = dc;

    block[0] = dc << (3 - s->intra_dc_precision);

    av_dlog(s->avctx, "dc=%d\n", block[0]);

    mismatch = block[0] ^ 1;

    i = 0;

    if (s->intra_vlc_format)

        rl = &ff_rl_mpeg2;

    else

        rl = &ff_rl_mpeg1;



    {

        OPEN_READER(re, &s->gb);

        /* now quantify & encode AC coefficients */

        for (;;) {

            UPDATE_CACHE(re, &s->gb);

            GET_RL_VLC(level, run, re, &s->gb, rl->rl_vlc[0], TEX_VLC_BITS, 2, 0);



            if (level == 127) {

                break;

            } else if (level != 0) {

                i += run;

                j  = scantable[i];

                level = (level * qscale * quant_matrix[j]) >> 4;

                level = (level ^ SHOW_SBITS(re, &s->gb, 1)) - SHOW_SBITS(re, &s->gb, 1);

                LAST_SKIP_BITS(re, &s->gb, 1);

            } else {

                /* escape */

                run = SHOW_UBITS(re, &s->gb, 6) + 1; LAST_SKIP_BITS(re, &s->gb, 6);

                UPDATE_CACHE(re, &s->gb);

                level = SHOW_SBITS(re, &s->gb, 12); SKIP_BITS(re, &s->gb, 12);

                i += run;

                j  = scantable[i];

                if (level < 0) {

                    level = (-level * qscale * quant_matrix[j]) >> 4;

                    level = -level;

                } else {

                    level = (level * qscale * quant_matrix[j]) >> 4;

                }

            }

            if (i > 63) {

                av_log(s->avctx, AV_LOG_ERROR, "ac-tex damaged at %d %d\n", s->mb_x, s->mb_y);

                return -1;

            }



            mismatch ^= level;

            block[j]  = level;

        }

        CLOSE_READER(re, &s->gb);

    }

    block[63] ^= mismatch & 1;



    s->block_last_index[n] = i;

    return 0;

}
