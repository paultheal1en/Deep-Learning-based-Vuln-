static int decode_p_mbs(VC9Context *v)

{

    MpegEncContext *s = &v->s;

    GetBitContext *gb = &v->s.gb;

    int current_mb = 0, i; /* MB/Block Position info */

    uint8_t cbpcy[4], previous_cbpcy[4], predicted_cbpcy,

        *p_cbpcy /* Pointer to skip some math */;

    int hybrid_pred; /* Prediction types */

    int mv_mode_bit = 0; 

    int mqdiff, mquant; /* MB quantization */

    int ttmb; /* MB Transform type */



    static const int size_table[6] = { 0, 2, 3, 4, 5, 8 },

        offset_table[6] = { 0, 1, 3, 7, 15, 31 };

        int mb_has_coeffs = 1; /* last_flag */

    int dmv_x, dmv_y; /* Differential MV components */

    int k_x, k_y; /* Long MV fixed bitlength */

    int hpel_flag; /* Some MB properties */

    int index, index1; /* LUT indices */

    int val, sign; /* MVDATA temp values */



    /* Select ttmb table depending on pq */

    if (v->pq < 5) v->ttmb_vlc = &vc9_ttmb_vlc[0];

    else if (v->pq < 13) v->ttmb_vlc = &vc9_ttmb_vlc[1];

    else v->ttmb_vlc = &vc9_ttmb_vlc[2];



    /* Select proper long MV range */

    switch (v->mvrange)

    {

    case 1: k_x = 10; k_y = 9; break;

    case 2: k_x = 12; k_y = 10; break;

    case 3: k_x = 13; k_y = 11; break;

    default: /*case 0 too */ k_x = 9; k_y = 8; break;

    }



    hpel_flag = v->mv_mode & 1; //MV_PMODE is HPEL

    k_x -= hpel_flag;

    k_y -= hpel_flag;



    /* Reset CBPCY predictors */

    memset(v->previous_line_cbpcy, 0, s->mb_stride<<2);



    for (s->mb_y=0; s->mb_y<s->mb_height; s->mb_y++)

    {

        /* Init CBPCY for line */

        *((uint32_t*)previous_cbpcy) = 0x00000000;

        p_cbpcy = v->previous_line_cbpcy+4;



        for (s->mb_x=0; s->mb_x<s->mb_width; s->mb_x++, p_cbpcy += 4)

        {

            if (v->mv_type_mb_plane.is_raw)

                v->mv_type_mb_plane.data[current_mb] = get_bits(gb, 1);

            if (v->skip_mb_plane.is_raw)

                v->skip_mb_plane.data[current_mb] = get_bits(gb, 1);

            if (!mv_mode_bit) /* 1MV mode */

            {

                if (!v->skip_mb_plane.data[current_mb])

                {

                    GET_MVDATA(dmv_x, dmv_y);



                    /* hybrid mv pred, 8.3.5.3.4 */

                    if (v->mv_mode == MV_PMODE_1MV ||

                        v->mv_mode == MV_PMODE_MIXED_MV)

                        hybrid_pred = get_bits(gb, 1);

                    if (s->mb_intra && !mb_has_coeffs)

                    {

                        GET_MQUANT();

                        s->ac_pred = get_bits(gb, 1);

                    }

                    else if (mb_has_coeffs)

                    {

                        if (s->mb_intra) s->ac_pred = get_bits(gb, 1);

                        predicted_cbpcy = get_vlc2(gb, v->cbpcy_vlc->table, VC9_CBPCY_P_VLC_BITS, 2);

                        cbpcy[0] = (p_cbpcy[-1] == p_cbpcy[2]) ? previous_cbpcy[1] : p_cbpcy[2];

                        cbpcy[0] ^= ((predicted_cbpcy>>5)&0x01);

                        cbpcy[1] = (p_cbpcy[2] == p_cbpcy[3]) ? cbpcy[0] : p_cbpcy[3];

                        cbpcy[1] ^= ((predicted_cbpcy>>4)&0x01);

                        cbpcy[2] = (previous_cbpcy[1] == cbpcy[0]) ? previous_cbpcy[3] : cbpcy[0];

                        cbpcy[2] ^= ((predicted_cbpcy>>3)&0x01);

                        cbpcy[3] = (cbpcy[1] == cbpcy[0]) ? cbpcy[2] : cbpcy[1];

                        cbpcy[3] ^= ((predicted_cbpcy>>2)&0x01);

                        //GET_CBPCY(v->cbpcy_vlc->table, VC9_CBPCY_P_VLC_BITS);



                        GET_MQUANT();

                    }

                    if (!v->ttmbf)

                        ttmb = get_vlc2(gb, v->ttmb_vlc->table,

                                            VC9_TTMB_VLC_BITS, 12);

                    /* TODO: decode blocks from that mb wrt cbpcy */

                }

                else //Skipped

                {

                    /* hybrid mv pred, 8.3.5.3.4 */

                    if (v->mv_mode == MV_PMODE_1MV ||

                        v->mv_mode == MV_PMODE_MIXED_MV)

                        hybrid_pred = get_bits(gb, 1);

                }

            } //1MV mode

            else //4MV mode

            {

              if (!v->skip_mb_plane.data[current_mb] /* unskipped MB */)

                {

                    /* Get CBPCY */

                    GET_CBPCY(v->cbpcy_vlc->table, VC9_CBPCY_P_VLC_BITS);

                    for (i=0; i<4; i++) //For all 4 Y blocks

                    {

                        if (cbpcy[i] /* cbpcy set for this block */)

                        {

                            GET_MVDATA(dmv_x, dmv_y);

                        }

                        if (v->mv_mode == MV_PMODE_MIXED_MV /* Hybrid pred */)

                            hybrid_pred = get_bits(gb, 1);

                        GET_MQUANT();

                        if (s->mb_intra /* One of the 4 blocks is intra */ &&

                            index /* non-zero pred for that block */)

                            s->ac_pred = get_bits(gb, 1);

                        if (!v->ttmbf)

                            ttmb = get_vlc2(gb, v->ttmb_vlc->table,

                                            VC9_TTMB_VLC_BITS, 12);

            

                        /* TODO: Process blocks wrt cbpcy */

            

                    }

                }

                else //Skipped MB

                {

                    for (i=0; i<4; i++) //All 4 Y blocks

                    {

                        if (v->mv_mode == MV_PMODE_MIXED_MV /* Hybrid pred */)

                            hybrid_pred = get_bits(gb, 1);

                        

                        /* TODO: do something */

                    }

                }

            }



            /* Update for next block */

#if TRACE > 2

            av_log(s->avctx, AV_LOG_DEBUG, "Block %4i: p_cbpcy=%i%i%i%i, previous_cbpcy=%i%i%i%i,"

                   " cbpcy=%i%i%i%i\n", current_mb,

                   p_cbpcy[0], p_cbpcy[1], p_cbpcy[2], p_cbpcy[3],

                   previous_cbpcy[0], previous_cbpcy[1], previous_cbpcy[2], previous_cbpcy[3],

                   cbpcy[0], cbpcy[1], cbpcy[2], cbpcy[3]);

#endif

            *((uint32_t*)p_cbpcy) = *((uint32_t*)previous_cbpcy);

            *((uint32_t*)previous_cbpcy) = *((uint32_t*)cbpcy);

            current_mb++;

        }

    }

    return 0;

}
