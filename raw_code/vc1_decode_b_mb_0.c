static void vc1_decode_b_mb(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    GetBitContext *gb = &s->gb;

    int i, j;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;

    int cbp = 0; /* cbp decoding stuff */

    int mqdiff, mquant; /* MB quantization */

    int ttmb = v->ttfrm; /* MB Transform type */

    int mb_has_coeffs = 0; /* last_flag */

    int index, index1; /* LUT indexes */

    int val, sign; /* temp values */

    int first_block = 1;

    int dst_idx, off;

    int skipped, direct;

    int dmv_x[2], dmv_y[2];

    int bmvtype = BMV_TYPE_BACKWARD;



    mquant = v->pq; /* Loosy initialization */

    s->mb_intra = 0;



    if (v->dmb_is_raw)

        direct = get_bits1(gb);

    else

        direct = v->direct_mb_plane[mb_pos];

    if (v->skip_is_raw)

        skipped = get_bits1(gb);

    else

        skipped = v->s.mbskip_table[mb_pos];



    dmv_x[0] = dmv_x[1] = dmv_y[0] = dmv_y[1] = 0;

    for(i = 0; i < 6; i++) {

        v->mb_type[0][s->block_index[i]] = 0;

        s->dc_val[0][s->block_index[i]] = 0;

    }

    s->current_picture.qscale_table[mb_pos] = 0;



    if (!direct) {

        if (!skipped) {

            GET_MVDATA(dmv_x[0], dmv_y[0]);

            dmv_x[1] = dmv_x[0];

            dmv_y[1] = dmv_y[0];

        }

        if(skipped || !s->mb_intra) {

            bmvtype = decode012(gb);

            switch(bmvtype) {

            case 0:

                bmvtype = (v->bfraction >= (B_FRACTION_DEN/2)) ? BMV_TYPE_BACKWARD : BMV_TYPE_FORWARD;

                break;

            case 1:

                bmvtype = (v->bfraction >= (B_FRACTION_DEN/2)) ? BMV_TYPE_FORWARD : BMV_TYPE_BACKWARD;

                break;

            case 2:

                bmvtype = BMV_TYPE_INTERPOLATED;

                dmv_x[0] = dmv_y[0] = 0;

            }

        }

    }

    for(i = 0; i < 6; i++)

        v->mb_type[0][s->block_index[i]] = s->mb_intra;



    if (skipped) {

        if(direct) bmvtype = BMV_TYPE_INTERPOLATED;

        vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype);

        vc1_b_mc(v, dmv_x, dmv_y, direct, bmvtype);

        return;

    }

    if (direct) {

        cbp = get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_CBPCY_P_VLC_BITS, 2);

        GET_MQUANT();

        s->mb_intra = 0;

        s->current_picture.qscale_table[mb_pos] = mquant;

        if(!v->ttmbf)

            ttmb = get_vlc2(gb, ff_vc1_ttmb_vlc[v->tt_index].table, VC1_TTMB_VLC_BITS, 2);

        dmv_x[0] = dmv_y[0] = dmv_x[1] = dmv_y[1] = 0;

        vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype);

        vc1_b_mc(v, dmv_x, dmv_y, direct, bmvtype);

    } else {

        if(!mb_has_coeffs && !s->mb_intra) {

            /* no coded blocks - effectively skipped */

            vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype);

            vc1_b_mc(v, dmv_x, dmv_y, direct, bmvtype);

            return;

        }

        if(s->mb_intra && !mb_has_coeffs) {

            GET_MQUANT();

            s->current_picture.qscale_table[mb_pos] = mquant;

            s->ac_pred = get_bits1(gb);

            cbp = 0;

            vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype);

        } else {

            if(bmvtype == BMV_TYPE_INTERPOLATED) {

                GET_MVDATA(dmv_x[0], dmv_y[0]);

                if(!mb_has_coeffs) {

                    /* interpolated skipped block */

                    vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype);

                    vc1_b_mc(v, dmv_x, dmv_y, direct, bmvtype);

                    return;

                }

            }

            vc1_pred_b_mv(v, dmv_x, dmv_y, direct, bmvtype);

            if(!s->mb_intra) {

                vc1_b_mc(v, dmv_x, dmv_y, direct, bmvtype);

            }

            if(s->mb_intra)

                s->ac_pred = get_bits1(gb);

            cbp = get_vlc2(&v->s.gb, v->cbpcy_vlc->table, VC1_CBPCY_P_VLC_BITS, 2);

            GET_MQUANT();

            s->current_picture.qscale_table[mb_pos] = mquant;

            if(!v->ttmbf && !s->mb_intra && mb_has_coeffs)

                ttmb = get_vlc2(gb, ff_vc1_ttmb_vlc[v->tt_index].table, VC1_TTMB_VLC_BITS, 2);

        }

    }

    dst_idx = 0;

    for (i=0; i<6; i++)

    {

        s->dc_val[0][s->block_index[i]] = 0;

        dst_idx += i >> 2;

        val = ((cbp >> (5 - i)) & 1);

        off = (i & 4) ? 0 : ((i & 1) * 8 + (i & 2) * 4 * s->linesize);

        v->mb_type[0][s->block_index[i]] = s->mb_intra;

        if(s->mb_intra) {

            /* check if prediction blocks A and C are available */

            v->a_avail = v->c_avail = 0;

            if(i == 2 || i == 3 || !s->first_slice_line)

                v->a_avail = v->mb_type[0][s->block_index[i] - s->block_wrap[i]];

            if(i == 1 || i == 3 || s->mb_x)

                v->c_avail = v->mb_type[0][s->block_index[i] - 1];



            vc1_decode_intra_block(v, s->block[i], i, val, mquant, (i&4)?v->codingset2:v->codingset);

            if((i>3) && (s->flags & CODEC_FLAG_GRAY)) continue;

            v->vc1dsp.vc1_inv_trans_8x8(s->block[i]);

            if(v->rangeredfrm) for(j = 0; j < 64; j++) s->block[i][j] <<= 1;

            s->dsp.put_signed_pixels_clamped(s->block[i], s->dest[dst_idx] + off, i & 4 ? s->uvlinesize : s->linesize);

        } else if(val) {

            vc1_decode_p_block(v, s->block[i], i, mquant, ttmb, first_block, s->dest[dst_idx] + off, (i&4)?s->uvlinesize:s->linesize, (i&4) && (s->flags & CODEC_FLAG_GRAY), 0, 0, 0);

            if(!v->ttmbf && ttmb < 8) ttmb = -1;

            first_block = 0;

        }

    }

}
