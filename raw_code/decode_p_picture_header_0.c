static int decode_p_picture_header(VC9Context *v)

{

    /* INTERFRM, FRMCNT, RANGEREDFRM read in caller */

    int lowquant, pqindex;



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

    av_log(v->avctx, AV_LOG_DEBUG, "P Frame: QP=%i (+%i/2)\n",

           v->pq, v->halfpq);

    if (v->extended_mv == 1) v->mvrange = get_prefix(&v->gb, 0, 3);

#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

    {

        if (v->postprocflag) v->postproc = get_bits(&v->gb, 1);

    }

    else

#endif

        if (v->multires) v->respic = get_bits(&v->gb, 2);

    lowquant = (v->pquantizer>12) ? 0 : 1;

    v->mv_mode = mv_pmode_table[lowquant][get_prefix(&v->gb, 1, 4)];

    if (v->mv_mode == MV_PMODE_INTENSITY_COMP)

    {

        v->mv_mode2 = mv_pmode_table[lowquant][get_prefix(&v->gb, 1, 3)];

        v->lumscale = get_bits(&v->gb, 6);

        v->lumshift = get_bits(&v->gb, 6);

    }



    if ((v->mv_mode == MV_PMODE_INTENSITY_COMP &&

         v->mv_mode2 == MV_PMODE_MIXED_MV)

        || v->mv_mode == MV_PMODE_MIXED_MV)

    {

        if (bitplane_decoding(v->mv_type_mb_plane, v->width_mb,

                                  v->height_mb, v) < 0)

            return -1;

    }



    if (bitplane_decoding(v->skip_mb_plane, v->width_mb,

                              v->height_mb, v) < 0)

        return -1;



    /* Hopefully this is correct for P frames */

    v->mv_diff_vlc = &vc9_mv_diff_vlc[get_bits(&v->gb, 2)];

    v->cbpcy_vlc = &vc9_cbpcy_p_vlc[get_bits(&v->gb, 2)];



    if (v->dquant)

    {

        av_log(v->avctx, AV_LOG_INFO, "VOP DQuant info\n");

        vop_dquant_decoding(v);

    }



    if (v->vstransform)

    {

        v->ttmbf = get_bits(&v->gb, 1);

        if (v->ttmbf)

        {

            v->ttfrm = get_bits(&v->gb, 2);

            av_log(v->avctx, AV_LOG_INFO, "Transform used: %ix%i\n",

                   (v->ttfrm & 2) ? 4 : 8, (v->ttfrm & 1) ? 4 : 8);

        }

    }

    /* Epilog should be done in caller */

    return 0;

}
