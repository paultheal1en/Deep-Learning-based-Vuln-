static int dv_encode_video_segment(AVCodecContext *avctx, DVwork_chunk *work_chunk)

{

    DVVideoContext *s = avctx->priv_data;

    int mb_index, i, j;

    int mb_x, mb_y, c_offset, linesize;

    uint8_t*  y_ptr;

    uint8_t*  data;

    uint8_t*  dif;

    uint8_t   scratch[64];

    EncBlockInfo  enc_blks[5*DV_MAX_BPM];

    PutBitContext pbs[5*DV_MAX_BPM];

    PutBitContext* pb;

    EncBlockInfo* enc_blk;

    int       vs_bit_size = 0;

    int       qnos[5] = {15, 15, 15, 15, 15}; /* No quantization */

    int*      qnosp = &qnos[0];



    dif = &s->buf[work_chunk->buf_offset*80];

    enc_blk = &enc_blks[0];

    for (mb_index = 0; mb_index < 5; mb_index++) {

        dv_calculate_mb_xy(s, work_chunk, mb_index, &mb_x, &mb_y);

        y_ptr    = s->picture.data[0] + ((mb_y * s->picture.linesize[0] + mb_x) << 3);

        c_offset = (((mb_y >>  (s->sys->pix_fmt == PIX_FMT_YUV420P)) * s->picture.linesize[1] +

                     (mb_x >> ((s->sys->pix_fmt == PIX_FMT_YUV411P) ? 2 : 1))) << 3);

        for (j = 0; j < 6; j++) {

            if (s->sys->pix_fmt == PIX_FMT_YUV422P) { /* 4:2:2 */

                if (j == 0 || j == 2) {

                    /* Y0 Y1 */

                    data     = y_ptr + ((j >> 1) * 8);

                    linesize = s->picture.linesize[0];

                } else if (j > 3) {

                    /* Cr Cb */

                    data     = s->picture.data[6 - j] + c_offset;

                    linesize = s->picture.linesize[6 - j];

                } else {

                    /* j=1 and j=3 are "dummy" blocks, used for AC data only */

                    data     = NULL;

                    linesize = 0;

                }

            } else { /* 4:1:1 or 4:2:0 */

                if (j < 4) {  /* Four Y blocks */

                    /* NOTE: at end of line, the macroblock is handled as 420 */

                    if (s->sys->pix_fmt == PIX_FMT_YUV411P && mb_x < (704 / 8)) {

                        data = y_ptr + (j * 8);

                    } else {

                        data = y_ptr + ((j & 1) * 8) + ((j >> 1) * 8 * s->picture.linesize[0]);

                    }

                    linesize = s->picture.linesize[0];

                } else {      /* Cr and Cb blocks */

                    /* don't ask Fabrice why they inverted Cb and Cr ! */

                    data     = s->picture.data    [6 - j] + c_offset;

                    linesize = s->picture.linesize[6 - j];

                    if (s->sys->pix_fmt == PIX_FMT_YUV411P && mb_x >= (704 / 8)) {

                        uint8_t* d;

                        uint8_t* b = scratch;

                        for (i = 0; i < 8; i++) {

                            d = data + 8 * linesize;

                            b[0] = data[0]; b[1] = data[1]; b[2] = data[2]; b[3] = data[3];

                            b[4] =    d[0]; b[5] =    d[1]; b[6] =    d[2]; b[7] =    d[3];

                            data += linesize;

                            b += 8;

                        }

                        data = scratch;

                        linesize = 8;

                    }

                }

            }



            vs_bit_size += dv_init_enc_block(enc_blk, data, linesize, s, j>>2);



            ++enc_blk;

        }

    }



    if (vs_total_ac_bits < vs_bit_size)

        dv_guess_qnos(&enc_blks[0], qnosp);



    /* DIF encoding process */

    for (j=0; j<5*s->sys->bpm;) {

        int start_mb = j;



        dif[3] = *qnosp++;

        dif += 4;



        /* First pass over individual cells only */

        for (i=0; i<s->sys->bpm; i++, j++) {

            int sz = s->sys->block_sizes[i]>>3;



            init_put_bits(&pbs[j], dif, sz);

            put_bits(&pbs[j], 9, (uint16_t)(((enc_blks[j].mb[0] >> 3) - 1024 + 2) >> 2));

            put_bits(&pbs[j], 1, enc_blks[j].dct_mode);

            put_bits(&pbs[j], 2, enc_blks[j].cno);



            dv_encode_ac(&enc_blks[j], &pbs[j], &pbs[j+1]);

            dif += sz;

        }



        /* Second pass over each MB space */

        pb = &pbs[start_mb];

        for (i=0; i<s->sys->bpm; i++) {

            if (enc_blks[start_mb+i].partial_bit_count)

                pb = dv_encode_ac(&enc_blks[start_mb+i], pb, &pbs[start_mb+s->sys->bpm]);

        }

    }



    /* Third and final pass over the whole video segment space */

    pb = &pbs[0];

    for (j=0; j<5*s->sys->bpm; j++) {

       if (enc_blks[j].partial_bit_count)

           pb = dv_encode_ac(&enc_blks[j], pb, &pbs[s->sys->bpm*5]);

       if (enc_blks[j].partial_bit_count)

            av_log(NULL, AV_LOG_ERROR, "ac bitstream overflow\n");

    }



    for (j=0; j<5*s->sys->bpm; j++)

       flush_put_bits(&pbs[j]);



    return 0;

}
