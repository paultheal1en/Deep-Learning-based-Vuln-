void h263_encode_picture_header(MpegEncContext * s, int picture_number)

{

    int format;



    align_put_bits(&s->pb);



    /* Update the pointer to last GOB */

    s->ptr_lastgob = pbBufPtr(&s->pb);

    s->gob_number = 0;



    put_bits(&s->pb, 22, 0x20); /* PSC */

    put_bits(&s->pb, 8, (((int64_t)s->picture_number * 30 * s->avctx->frame_rate_base) / 

                         s->avctx->frame_rate) & 0xff);



    put_bits(&s->pb, 1, 1);	/* marker */

    put_bits(&s->pb, 1, 0);	/* h263 id */

    put_bits(&s->pb, 1, 0);	/* split screen off */

    put_bits(&s->pb, 1, 0);	/* camera  off */

    put_bits(&s->pb, 1, 0);	/* freeze picture release off */

    

    format = h263_get_picture_format(s->width, s->height);

    if (!s->h263_plus) {

        /* H.263v1 */

        put_bits(&s->pb, 3, format);

        put_bits(&s->pb, 1, (s->pict_type == P_TYPE));

        /* By now UMV IS DISABLED ON H.263v1, since the restrictions

        of H.263v1 UMV implies to check the predicted MV after

        calculation of the current MB to see if we're on the limits */

        put_bits(&s->pb, 1, 0);	/* unrestricted motion vector: off */

        put_bits(&s->pb, 1, 0);	/* SAC: off */

        put_bits(&s->pb, 1, s->obmc);	/* advanced prediction mode */

        put_bits(&s->pb, 1, 0);	/* not PB frame */

        put_bits(&s->pb, 5, s->qscale);

        put_bits(&s->pb, 1, 0);	/* Continuous Presence Multipoint mode: off */

    } else {

        /* H.263v2 */

        /* H.263 Plus PTYPE */

        put_bits(&s->pb, 3, 7);

        put_bits(&s->pb,3,1); /* Update Full Extended PTYPE */

        if (format == 7)

            put_bits(&s->pb,3,6); /* Custom Source Format */

        else

            put_bits(&s->pb, 3, format);

            

        put_bits(&s->pb,1,0); /* Custom PCF: off */

        s->umvplus = s->unrestricted_mv;

        put_bits(&s->pb, 1, s->umvplus); /* Unrestricted Motion Vector */

        put_bits(&s->pb,1,0); /* SAC: off */

        put_bits(&s->pb,1,s->obmc); /* Advanced Prediction Mode */

        put_bits(&s->pb,1,s->h263_aic); /* Advanced Intra Coding */

        put_bits(&s->pb,1,0); /* Deblocking Filter: off */

        put_bits(&s->pb,1,0); /* Slice Structured: off */

        put_bits(&s->pb,1,0); /* Reference Picture Selection: off */

        put_bits(&s->pb,1,0); /* Independent Segment Decoding: off */

        put_bits(&s->pb,1,s->alt_inter_vlc); /* Alternative Inter VLC */

        put_bits(&s->pb,1,0); /* Modified Quantization: off */

        put_bits(&s->pb,1,1); /* "1" to prevent start code emulation */

        put_bits(&s->pb,3,0); /* Reserved */

		

        put_bits(&s->pb, 3, s->pict_type == P_TYPE);

		

        put_bits(&s->pb,1,0); /* Reference Picture Resampling: off */

        put_bits(&s->pb,1,0); /* Reduced-Resolution Update: off */

        put_bits(&s->pb,1,s->no_rounding); /* Rounding Type */

        put_bits(&s->pb,2,0); /* Reserved */

        put_bits(&s->pb,1,1); /* "1" to prevent start code emulation */

		

        /* This should be here if PLUSPTYPE */

        put_bits(&s->pb, 1, 0);	/* Continuous Presence Multipoint mode: off */

		

		if (format == 7) {

            /* Custom Picture Format (CPFMT) */

            aspect_to_info(s, s->avctx->sample_aspect_ratio);



            put_bits(&s->pb,4,s->aspect_ratio_info);

            put_bits(&s->pb,9,(s->width >> 2) - 1);

            put_bits(&s->pb,1,1); /* "1" to prevent start code emulation */

            put_bits(&s->pb,9,(s->height >> 2));

            if (s->aspect_ratio_info == FF_ASPECT_EXTENDED){

                put_bits(&s->pb, 8, s->avctx->sample_aspect_ratio.num);

                put_bits(&s->pb, 8, s->avctx->sample_aspect_ratio.den);

	    }

        }

        

        /* Unlimited Unrestricted Motion Vectors Indicator (UUI) */

        if (s->umvplus)

//            put_bits(&s->pb,1,1); /* Limited according tables of Annex D */

            put_bits(&s->pb,2,1); /* unlimited */



        put_bits(&s->pb, 5, s->qscale);

    }



    put_bits(&s->pb, 1, 0);	/* no PEI */



    if(s->h263_aic){

         s->y_dc_scale_table= 

         s->c_dc_scale_table= ff_aic_dc_scale_table;

    }else{

        s->y_dc_scale_table=

        s->c_dc_scale_table= ff_mpeg1_dc_scale_table;

    }

}
