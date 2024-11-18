static void vc1_decode_i_blocks_adv(VC1Context *v)

{

    int k, j;

    MpegEncContext *s = &v->s;

    int cbp, val;

    uint8_t *coded_val;

    int mb_pos;

    int mquant = v->pq;

    int mqdiff;

    int overlap;

    GetBitContext *gb = &s->gb;



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

    ff_er_add_slice(s, 0, 0, s->mb_width - 1, s->mb_height - 1, (AC_END|DC_END|MV_END));

    for(s->mb_y = 0; s->mb_y < s->mb_height; s->mb_y++) {

        for(s->mb_x = 0; s->mb_x < s->mb_width; s->mb_x++) {

            ff_init_block_index(s);

            ff_update_block_index(s);

            s->dsp.clear_blocks(s->block[0]);

            mb_pos = s->mb_x + s->mb_y * s->mb_stride;

            s->current_picture.mb_type[mb_pos] = MB_TYPE_INTRA;

            s->current_picture.motion_val[1][s->block_index[0]][0] = 0;

            s->current_picture.motion_val[1][s->block_index[0]][1] = 0;



            // do actual MB decoding and displaying

            cbp = get_vlc2(&v->s.gb, ff_msmp4_mb_i_vlc.table, MB_INTRA_VLC_BITS, 2);

            if(v->acpred_is_raw)

                v->s.ac_pred = get_bits(&v->s.gb, 1);

            else

                v->s.ac_pred = v->acpred_plane[mb_pos];



            if(v->condover == CONDOVER_SELECT) {

                if(v->overflg_is_raw)

                    overlap = get_bits(&v->s.gb, 1);

                else

                    overlap = v->over_flags_plane[mb_pos];

            } else

                overlap = (v->condover == CONDOVER_ALL);



            GET_MQUANT();



            s->current_picture.qscale_table[mb_pos] = mquant;



            for(k = 0; k < 6; k++) {

                val = ((cbp >> (5 - k)) & 1);



                if (k < 4) {

                    int pred = vc1_coded_block_pred(&v->s, k, &coded_val);

                    val = val ^ pred;

                    *coded_val = val;

                }

                cbp |= val << (5 - k);



                v->a_avail = !s->first_slice_line || (k==2 || k==3);

                v->c_avail = !!s->mb_x || (k==1 || k==3);



                vc1_decode_i_block_adv(v, s->block[k], k, val, (k<4)? v->codingset : v->codingset2, mquant);



                s->dsp.vc1_inv_trans_8x8(s->block[k]);

                for(j = 0; j < 64; j++) s->block[k][j] += 128;

            }



            vc1_put_block(v, s->block);

            if(overlap) {

                if(s->mb_x) {

                    s->dsp.vc1_h_overlap(s->dest[0], s->linesize, 0);

                    s->dsp.vc1_h_overlap(s->dest[0] + 8 * s->linesize, s->linesize, 0);

                    if(!(s->flags & CODEC_FLAG_GRAY)) {

                        s->dsp.vc1_h_overlap(s->dest[1], s->uvlinesize, s->mb_x&1);

                        s->dsp.vc1_h_overlap(s->dest[2], s->uvlinesize, s->mb_x&1);

                    }

                }

                s->dsp.vc1_h_overlap(s->dest[0] + 8, s->linesize, 1);

                s->dsp.vc1_h_overlap(s->dest[0] + 8 * s->linesize + 8, s->linesize, 1);

                if(!s->first_slice_line) {

                    s->dsp.vc1_v_overlap(s->dest[0], s->linesize, 0);

                    s->dsp.vc1_v_overlap(s->dest[0] + 8, s->linesize, 0);

                    if(!(s->flags & CODEC_FLAG_GRAY)) {

                        s->dsp.vc1_v_overlap(s->dest[1], s->uvlinesize, s->mb_y&1);

                        s->dsp.vc1_v_overlap(s->dest[2], s->uvlinesize, s->mb_y&1);

                    }

                }

                s->dsp.vc1_v_overlap(s->dest[0] + 8 * s->linesize, s->linesize, 1);

                s->dsp.vc1_v_overlap(s->dest[0] + 8 * s->linesize + 8, s->linesize, 1);

            }



            if(get_bits_count(&s->gb) > v->bits) {

                av_log(s->avctx, AV_LOG_ERROR, "Bits overconsumption: %i > %i\n", get_bits_count(&s->gb), v->bits);

                return;

            }

        }

        ff_draw_horiz_band(s, s->mb_y * 16, 16);

        s->first_slice_line = 0;

    }

}
