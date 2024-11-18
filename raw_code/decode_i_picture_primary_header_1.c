static int decode_i_picture_primary_header(VC9Context *v)

{

    GetBitContext *gb = &v->s.gb;

    int pqindex;



    /* Prolog common to all frametypes should be done in caller */

    //BF = Buffer Fullness

    if (v->profile <= PROFILE_MAIN && get_bits(gb, 7))

    {

        av_log(v->s.avctx, AV_LOG_DEBUG, "I BufferFullness not 0\n");

    }



    /* Quantizer stuff */

    pqindex = get_bits(gb, 5);

    if (v->quantizer_mode == QUANT_FRAME_IMPLICIT)

        v->pq = pquant_table[0][pqindex];

    else

    {

        v->pq = pquant_table[v->quantizer_mode-1][pqindex];

    }

    if (pqindex < 9) v->halfpq = get_bits(gb, 1);

    if (v->quantizer_mode == QUANT_FRAME_EXPLICIT)

        v->pquantizer = get_bits(gb, 1);

    av_log(v->s.avctx, AV_LOG_DEBUG, "I frame: QP=%i (+%i/2)\n",

           v->pq, v->halfpq);

    return 0;

}
