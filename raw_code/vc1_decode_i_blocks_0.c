static void vc1_decode_i_blocks(VC1Context *v)

{

    int k, j;

    MpegEncContext *s = &v->s;

    int cbp, val;

    uint8_t *coded_val;

    int mb_pos;



    /* select codingmode used for VLC tables selection */

    switch(v->y_ac_table_index){

    case 0:

        v->codingset = (v->pqindex <= 8) ? CS_HIGH_RATE_INTRA : CS_LOW_MOT_INTRA;

        break;

    case 1:

        v->codingset = CS_HIGH_MOT_INTRA;

        break;

    case 2:

        v->codingset = CS_MID_RATE_INTRA;

        break;

    }



    switch(v->c_ac_table_index){

    case 0:

        v->codingset2 = (v->pqindex <= 8) ? CS_HIGH_RATE_INTER : CS_LOW_MOT_INTER;

        break;

    case 1:

        v->codingset2 = CS_HIGH_MOT_INTER;

        break;

    case 2:

        v->codingset2 = CS_MID_RATE_INTER;

        break;

    }



    /* Set DC scale - y and c use the same */

    s->y_dc_scale = s->y_dc_scale_table[v->pq];

    s->c_dc_scale = s->c_dc_scale_table[v->pq];



    //do frame decode

    s->mb_x = s->mb_y = 0;

    s->mb_intra = 1;

    s->first_slice_line = 1;

    for(s->mb_y = 0; s->mb_y < s->mb_height; s->mb_y++) {

        for(s->mb_x = 0; s->mb_x < s->mb_width; s->mb_x++) {

            ff_init_block_index(s);

            ff_update_block_index(s);

            s->dsp.clear_blocks(s->block[0]);

            mb_pos = s->mb_x + s->mb_y * s->mb_width;

            s->current_picture.mb_type[mb_pos] = MB_TYPE_INTRA;

            s->current_picture.qscale_table[mb_pos] = v->pq;

            s->current_picture.motion_val[1][s->block_index[0]][0] = 0;

            s->current_picture.motion_val[1][s->block_index[0]][1] = 0;



            // do actual MB decoding and displaying

            cbp = get_vlc2(&v->s.gb, ff_msmp4_mb_i_vlc.table, MB_INTRA_VLC_BITS, 2);

            v->s.ac_pred = get_bits1(&v->s.gb);



            for(k = 0; k < 6; k++) {

                val = ((cbp >> (5 - k)) & 1);



                if (k < 4) {

                    int pred = vc1_coded_block_pred(&v->s, k, &coded_val);

                    val = val ^ pred;

                    *coded_val = val;

                }

                cbp |= val << (5 - k);



                vc1_decode_i_block(v, s->block[k], k, val, (k<4)? v->codingset : v->codingset2);



                s->dsp.vc1_inv_trans_8x8(s->block[k]);

                if(v->pq >= 9 && v->overlap) {

                    for(j = 0; j < 64; j++) s->block[k][j] += 128;

                }

            }



            vc1_put_block(v, s->block);

            if(v->pq >= 9 && v->overlap) {

                if(s->mb_x) {

                    s->dsp.vc1_h_overlap(s->dest[0], s->linesize);

                    s->dsp.vc1_h_overlap(s->dest[0] + 8 * s->linesize, s->linesize);

                    if(!(s->flags & CODEC_FLAG_GRAY)) {

                        s->dsp.vc1_h_overlap(s->dest[1], s->uvlinesize);

                        s->dsp.vc1_h_overlap(s->dest[2], s->uvlinesize);

                    }

                }

                s->dsp.vc1_h_overlap(s->dest[0] + 8, s->linesize);

                s->dsp.vc1_h_overlap(s->dest[0] + 8 * s->linesize + 8, s->linesize);

                if(!s->first_slice_line) {

                    s->dsp.vc1_v_overlap(s->dest[0], s->linesize);

                    s->dsp.vc1_v_overlap(s->dest[0] + 8, s->linesize);

                    if(!(s->flags & CODEC_FLAG_GRAY)) {

                        s->dsp.vc1_v_overlap(s->dest[1], s->uvlinesize);

                        s->dsp.vc1_v_overlap(s->dest[2], s->uvlinesize);

                    }

                }

                s->dsp.vc1_v_overlap(s->dest[0] + 8 * s->linesize, s->linesize);

                s->dsp.vc1_v_overlap(s->dest[0] + 8 * s->linesize + 8, s->linesize);

            }

            if(v->s.loop_filter) vc1_loop_filter_iblk(s, s->current_picture.qscale_table[mb_pos]);



            if(get_bits_count(&s->gb) > v->bits) {

                ff_er_add_slice(s, 0, 0, s->mb_x, s->mb_y, (AC_END|DC_END|MV_END));

                av_log(s->avctx, AV_LOG_ERROR, "Bits overconsumption: %i > %i\n", get_bits_count(&s->gb), v->bits);

                return;

            }

        }

        ff_draw_horiz_band(s, s->mb_y * 16, 16);

        s->first_slice_line = 0;

    }

    ff_er_add_slice(s, 0, 0, s->mb_width - 1, s->mb_height - 1, (AC_END|DC_END|MV_END));

}
