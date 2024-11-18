static int standard_decode_i_mbs(VC9Context *v)

{

    GetBitContext *gb = &v->s.gb;

    MpegEncContext *s = &v->s;

    int current_mb = 0; /* MB/Block Position info */

    uint8_t cbpcy[4], previous_cbpcy[4], predicted_cbpcy,

        *p_cbpcy /* Pointer to skip some math */;



    /* Reset CBPCY predictors */

    memset(v->previous_line_cbpcy, 0, s->mb_stride<<2);



    /* Select ttmb table depending on pq */

    if (v->pq < 5) v->ttmb_vlc = &vc9_ttmb_vlc[0];

    else if (v->pq < 13) v->ttmb_vlc = &vc9_ttmb_vlc[1];

    else v->ttmb_vlc = &vc9_ttmb_vlc[2];



    for (s->mb_y=0; s->mb_y<s->mb_height; s->mb_y++)

    {

        /* Init CBPCY for line */

        *((uint32_t*)previous_cbpcy) = 0x00000000;

        p_cbpcy = v->previous_line_cbpcy+4;



        for (s->mb_x=0; s->mb_x<s->mb_width; s->mb_x++, p_cbpcy += 4)

        {

            /* Get CBPCY */

            GET_CBPCY(ff_msmp4_mb_i_vlc.table, MB_INTRA_VLC_BITS);



            s->ac_pred = get_bits(gb, 1);



            /* TODO: Decode blocks from that mb wrt cbpcy */



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
