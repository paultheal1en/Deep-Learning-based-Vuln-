static int mjpeg_decode_com(MJpegDecodeContext *s)

{

    int i;

    UINT8 *cbuf;



    /* XXX: verify len field validity */

    unsigned int len = get_bits(&s->gb, 16)-2;

    cbuf = av_malloc(len+1);



    for (i = 0; i < len; i++)

	cbuf[i] = get_bits(&s->gb, 8);

    if (cbuf[i-1] == '\n')

	cbuf[i-1] = 0;

    else

	cbuf[i] = 0;



    printf("mjpeg comment: '%s'\n", cbuf);



    /* buggy avid, it puts EOI only at every 10th frame */

    if (!strcmp(cbuf, "AVID"))

    {

	s->buggy_avid = 1;

//	if (s->first_picture)

//	    printf("mjpeg: workarounding buggy AVID\n");

    }

    

    av_free(cbuf);



    return 0;

}
