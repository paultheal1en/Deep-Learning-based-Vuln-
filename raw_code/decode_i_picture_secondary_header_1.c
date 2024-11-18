static int decode_i_picture_secondary_header(VC9Context *v)

{

    int status;

#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

    {

        v->s.ac_pred = get_bits(&v->s.gb, 1);

        if (v->postprocflag) v->postproc = get_bits(&v->s.gb, 1);

        /* 7.1.1.34 + 8.5.2 */

        if (v->overlap && v->pq<9)

        {

            v->condover = get_bits(&v->s.gb, 1);

            if (v->condover)

            {

                v->condover = 2+get_bits(&v->s.gb, 1);

                if (v->condover == 3)

                {

                    status = bitplane_decoding(&v->over_flags_plane, v);

                    if (status < 0) return -1;

#  if TRACE

                    av_log(v->s.avctx, AV_LOG_DEBUG, "Overflags plane encoding: "

                           "Imode: %i, Invert: %i\n", status>>1, status&1);

#  endif

                }

            }

        }

    }

#endif



    /* Epilog (AC/DC syntax) should be done in caller */

    return 0;

}
