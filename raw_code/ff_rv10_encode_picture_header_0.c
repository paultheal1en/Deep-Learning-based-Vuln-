void ff_rv10_encode_picture_header(MpegEncContext *s, int picture_number)

{

    int full_frame= 0;



    avpriv_align_put_bits(&s->pb);



    put_bits(&s->pb, 1, 1);     /* marker */



    put_bits(&s->pb, 1, (s->pict_type == AV_PICTURE_TYPE_P));



    put_bits(&s->pb, 1, 0);     /* not PB frame */



    put_bits(&s->pb, 5, s->qscale);



    if (s->pict_type == AV_PICTURE_TYPE_I) {

        /* specific MPEG like DC coding not used */

    }

    /* if multiple packets per frame are sent, the position at which

       to display the macroblocks is coded here */

    if(!full_frame){

        put_bits(&s->pb, 6, 0); /* mb_x */

        put_bits(&s->pb, 6, 0); /* mb_y */

        put_bits(&s->pb, 12, s->mb_width * s->mb_height);

    }



    put_bits(&s->pb, 3, 0);     /* ignored */

}
