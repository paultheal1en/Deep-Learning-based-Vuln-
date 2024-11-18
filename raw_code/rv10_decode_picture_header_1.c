static int rv10_decode_picture_header(MpegEncContext *s)

{

    int mb_count, pb_frame, marker, h, full_frame;

    

    /* skip packet header */

    h = get_bits(&s->gb, 8);

    if ((h & 0xc0) == 0xc0) {

        int len, pos;

        full_frame = 1;

        len = get_num(&s->gb);

        pos = get_num(&s->gb);

    } else {

        int seq, frame_size, pos;

        full_frame = 0;

        seq = get_bits(&s->gb, 8);

        frame_size = get_num(&s->gb);

        pos = get_num(&s->gb);

    }

    /* picture number */

    get_bits(&s->gb, 8);



    marker = get_bits(&s->gb, 1);



    if (get_bits(&s->gb, 1))

        s->pict_type = P_TYPE;

    else

        s->pict_type = I_TYPE;



    pb_frame = get_bits(&s->gb, 1);



#ifdef DEBUG

    printf("pict_type=%d pb_frame=%d\n", s->pict_type, pb_frame);

#endif

    

    if (pb_frame)

        return -1;



    s->qscale = get_bits(&s->gb, 5);



    if (s->pict_type == I_TYPE) {

        if (s->rv10_version == 3) {

            /* specific MPEG like DC coding not used */

            s->last_dc[0] = get_bits(&s->gb, 8);

            s->last_dc[1] = get_bits(&s->gb, 8);

            s->last_dc[2] = get_bits(&s->gb, 8);

#ifdef DEBUG

            printf("DC:%d %d %d\n",

                   s->last_dc[0],

                   s->last_dc[1],

                   s->last_dc[2]);

#endif

        }

    }

    /* if multiple packets per frame are sent, the position at which

       to display the macro blocks is coded here */

    if (!full_frame) {

        s->mb_x = get_bits(&s->gb, 6);	/* mb_x */

        s->mb_y = get_bits(&s->gb, 6);	/* mb_y */

        mb_count = get_bits(&s->gb, 12);

    } else {

        s->mb_x = 0;

        s->mb_y = 0;

        mb_count = s->mb_width * s->mb_height;

    }



    get_bits(&s->gb, 3);	/* ignored */

    s->f_code = 1;

    s->unrestricted_mv = 1;

#if 0

    s->h263_long_vectors = 1;

#endif

    return mb_count;

}
