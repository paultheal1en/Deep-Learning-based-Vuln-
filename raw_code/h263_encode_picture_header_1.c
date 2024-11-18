void h263_encode_picture_header(MpegEncContext * s, int picture_number)

{

    int format;



    align_put_bits(&s->pb);

    put_bits(&s->pb, 22, 0x20);

    put_bits(&s->pb, 8, ((s->picture_number * 30 * FRAME_RATE_BASE) / 

                         s->frame_rate) & 0xff);



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

        put_bits(&s->pb, 1, 0);	/* advanced prediction mode: off */

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

        umvplus = (s->pict_type == P_TYPE) && s->unrestricted_mv;

        put_bits(&s->pb, 1, umvplus); /* Unrestricted Motion Vector */

        put_bits(&s->pb,1,0); /* SAC: off */

        put_bits(&s->pb,1,0); /* Advanced Prediction Mode: off */

        put_bits(&s->pb,1,0); /* Advanced Intra Coding: off */

        put_bits(&s->pb,1,0); /* Deblocking Filter: off */

        put_bits(&s->pb,1,0); /* Slice Structured: off */

        put_bits(&s->pb,1,0); /* Reference Picture Selection: off */

        put_bits(&s->pb,1,0); /* Independent Segment Decoding: off */

        put_bits(&s->pb,1,0); /* Alternative Inter VLC: off */

        put_bits(&s->pb,1,0); /* Modified Quantization: off */

        put_bits(&s->pb,1,1); /* "1" to prevent start code emulation */

        put_bits(&s->pb,3,0); /* Reserved */

		

        put_bits(&s->pb, 3, s->pict_type == P_TYPE);

		

        put_bits(&s->pb,1,0); /* Reference Picture Resampling: off */

        put_bits(&s->pb,1,0); /* Reduced-Resolution Update: off */

        put_bits(&s->pb,1,0); /* Rounding Type */

        put_bits(&s->pb,2,0); /* Reserved */

        put_bits(&s->pb,1,1); /* "1" to prevent start code emulation */

		

        /* This should be here if PLUSPTYPE */

        put_bits(&s->pb, 1, 0);	/* Continuous Presence Multipoint mode: off */

		

		if (format == 7) {

            /* Custom Picture Format (CPFMT) */

		

            put_bits(&s->pb,4,2); /* Aspect ratio: CIF 12:11 (4:3) picture */

            put_bits(&s->pb,9,(s->width >> 2) - 1);

            put_bits(&s->pb,1,1); /* "1" to prevent start code emulation */

            put_bits(&s->pb,9,(s->height >> 2));

        }

        

        /* Unlimited Unrestricted Motion Vectors Indicator (UUI) */

        if (umvplus)

            put_bits(&s->pb,1,1); /* Limited according tables of Annex D */

        put_bits(&s->pb, 5, s->qscale);

    }



    put_bits(&s->pb, 1, 0);	/* no PEI */

}
