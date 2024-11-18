static int standard_decode_picture_primary_header(VC9Context *v)

{

    GetBitContext *gb = &v->s.gb;

    int status = 0;



    if (v->finterpflag) v->interpfrm = get_bits(gb, 1);

    skip_bits(gb, 2); //framecnt unused

    if (v->rangered) v->rangeredfrm = get_bits(gb, 1);

    v->s.pict_type = get_bits(gb, 1);

    if (v->s.avctx->max_b_frames)

    {

        if (!v->s.pict_type)

        {

            if (get_bits(gb, 1)) v->s.pict_type = I_TYPE;

            else v->s.pict_type = B_TYPE;

        }

        else v->s.pict_type = P_TYPE;

    }

    else v->s.pict_type++;



    switch (v->s.pict_type)

    {

    case I_TYPE: status = decode_i_picture_header(v); break;

    case P_TYPE: status = decode_p_picture_primary_header(v); break;

    case BI_TYPE:

    case B_TYPE: status = decode_b_picture_primary_header(v); break;

    }



    if (status == FRAME_SKIPED)

    {

      av_log(v->s.avctx, AV_LOG_INFO, "Skipping frame...\n");

      return status;

    }

    return 0;

}
