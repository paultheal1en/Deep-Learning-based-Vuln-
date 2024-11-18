static int decode_i_picture_header(VC9Context *v)

{

  int pqindex, status = 0, ac_pred, condover;



    /* Prolog common to all frametypes should be done in caller */

    //BF = Buffer Fullness

    if (v->profile <= PROFILE_MAIN && get_bits(&v->gb, 7))

    {

        av_log(v, AV_LOG_DEBUG, "I BufferFullness not 0\n");

    }



    /* Quantizer stuff */

    pqindex = get_bits(&v->gb, 5);

    if (v->quantizer_mode == QUANT_FRAME_IMPLICIT)

        v->pq = pquant_table[0][pqindex];

    else

    {

        v->pq = pquant_table[v->quantizer_mode-1][pqindex];

    }

    if (pqindex < 9) v->halfpq = get_bits(&v->gb, 1);

    if (v->quantizer_mode == QUANT_FRAME_EXPLICIT)

        v->pquantizer = get_bits(&v->gb, 1);

    av_log(v->avctx, AV_LOG_DEBUG, "I frame: QP=%i (+%i/2)\n",

           v->pq, v->halfpq);

#if HAS_ADVANCED_PROFILE

    if (v->profile <= PROFILE_MAIN)

#endif

    {

        if (v->extended_mv) v->mvrange = get_prefix(&v->gb, 0, 3);

        if (v->multires) v->respic = get_bits(&v->gb, 2);

    }

#if HAS_ADVANCED_PROFILE

    else

    {

        ac_pred = get_bits(&v->gb, 1);

        if (v->postprocflag) v->postproc = get_bits(&v->gb, 1);

        /* 7.1.1.34 + 8.5.2 */

        if (v->overlap && v->pq<9)

        {

            condover = get_bits(&v->gb, 1);

            if (condover)

            {

                condover = 2+get_bits(&v->gb, 1);

                if (condover == 3)

                    status = bitplane_decoding(v->over_flags_plane,

                                                   v->width_mb, v->height_mb, v);

            }

        }

    }

#endif



    /* Epilog should be done in caller */

    return status;

}
