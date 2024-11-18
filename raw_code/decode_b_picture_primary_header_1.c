static int decode_b_picture_primary_header(VC9Context *v)

{

    GetBitContext *gb = &v->s.gb;

    int pqindex;



    /* Prolog common to all frametypes should be done in caller */

    if (v->profile == PROFILE_SIMPLE)

    {

        av_log(v->s.avctx, AV_LOG_ERROR, "Found a B frame while in Simple Profile!\n");

        return FRAME_SKIPED;

    }

    v->bfraction = vc9_bfraction_lut[get_vlc2(gb, vc9_bfraction_vlc.table,

                                              VC9_BFRACTION_VLC_BITS, 2)];

    if (v->bfraction < -1)

    {

        av_log(v->s.avctx, AV_LOG_ERROR, "Invalid BFRaction\n");

        return FRAME_SKIPED;

    }

    else if (!v->bfraction)

    {

        /* We actually have a BI frame */

        v->s.pict_type = BI_TYPE;

        v->buffer_fullness = get_bits(gb, 7);

    }



    /* Read the quantization stuff */

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



    if (v->profile > PROFILE_MAIN)

    {

        if (v->postprocflag) v->postproc = get_bits(gb, 2);

        if (v->extended_mv == 1 && v->s.pict_type != BI_TYPE)

            v->mvrange = get_prefix(gb, 0, 3);

    }

    else

    {

        if (v->extended_mv == 1)

            v->mvrange = get_prefix(gb, 0, 3);

    }

    /* Read the MV mode */

    if (v->s.pict_type != BI_TYPE)

    {

        v->mv_mode = get_bits(gb, 1);

        if (v->pq < 13)

        {

            if (!v->mv_mode)

            {

                v->mv_mode = get_bits(gb, 2);

                if (v->mv_mode)

                av_log(v->s.avctx, AV_LOG_ERROR,

                       "mv_mode for lowquant B frame was %i\n", v->mv_mode);

            }

        }

        else

        {

            if (!v->mv_mode)

            {

                if (get_bits(gb, 1))

                     av_log(v->s.avctx, AV_LOG_ERROR,

                            "mv_mode for highquant B frame was %i\n", v->mv_mode);

            }

            v->mv_mode = 1-v->mv_mode; //To match (pq < 13) mapping

        }

    }



    return 0;

}
