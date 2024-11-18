static int mjpegb_decode_frame(AVCodecContext *avctx, 

                              void *data, int *data_size,

                              UINT8 *buf, int buf_size)

{

    MJpegDecodeContext *s = avctx->priv_data;

    UINT8 *buf_end, *buf_ptr;

    int i;

    AVPicture *picture = data;

    GetBitContext hgb; /* for the header */

    uint32_t dqt_offs, dht_offs, sof_offs, sos_offs, second_field_offs;

    uint32_t field_size;



    *data_size = 0;



    /* no supplementary picture */

    if (buf_size == 0)

        return 0;



    buf_ptr = buf;

    buf_end = buf + buf_size;

    

read_header:

    /* reset on every SOI */

    s->restart_interval = 0;



    init_get_bits(&hgb, buf_ptr, /*buf_size*/buf_end - buf_ptr);



    skip_bits(&hgb, 32); /* reserved zeros */

    

    if (get_bits(&hgb, 32) != be2me_32(ff_get_fourcc("mjpg")))

    {

	dprintf("not mjpeg-b (bad fourcc)\n");

	return 0;

    }



    field_size = get_bits(&hgb, 32); /* field size */

    dprintf("field size: 0x%x\n", field_size);

    skip_bits(&hgb, 32); /* padded field size */

    second_field_offs = get_bits(&hgb, 32);

    dprintf("second field offs: 0x%x\n", second_field_offs);

    if (second_field_offs)

	s->interlaced = 1;



    dqt_offs = get_bits(&hgb, 32);

    dprintf("dqt offs: 0x%x\n", dqt_offs);

    if (dqt_offs)

    {

	init_get_bits(&s->gb, buf+dqt_offs, buf_end - (buf+dqt_offs));

	s->start_code = DQT;

	mjpeg_decode_dqt(s);

    }

    

    dht_offs = get_bits(&hgb, 32);

    dprintf("dht offs: 0x%x\n", dht_offs);

    if (dht_offs)

    {

	init_get_bits(&s->gb, buf+dht_offs, buf_end - (buf+dht_offs));

	s->start_code = DHT;

	mjpeg_decode_dht(s);

    }



    sof_offs = get_bits(&hgb, 32);

    dprintf("sof offs: 0x%x\n", sof_offs);

    if (sof_offs)

    {

	init_get_bits(&s->gb, buf+sof_offs, buf_end - (buf+sof_offs));

	s->start_code = SOF0;

	if (mjpeg_decode_sof0(s) < 0)

	    return -1;

    }



    sos_offs = get_bits(&hgb, 32);

    dprintf("sos offs: 0x%x\n", sos_offs);

    if (sos_offs)

    {

//	init_get_bits(&s->gb, buf+sos_offs, buf_end - (buf+sos_offs));

	init_get_bits(&s->gb, buf+sos_offs, field_size);

	s->start_code = SOS;

	mjpeg_decode_sos(s);

    }



    skip_bits(&hgb, 32); /* start of data offset */



    if (s->interlaced) {

        s->bottom_field ^= 1;

        /* if not bottom field, do not output image yet */

        if (s->bottom_field && second_field_offs)

	{

	    buf_ptr = buf + second_field_offs;

	    second_field_offs = 0;

	    goto read_header;

    	}

    }



    for(i=0;i<3;i++) {

        picture->data[i] = s->current_picture[i];

        picture->linesize[i] = (s->interlaced) ?

    	    s->linesize[i] >> 1 : s->linesize[i];

    }

    *data_size = sizeof(AVPicture);

    avctx->height = s->height;

    if (s->interlaced)

        avctx->height *= 2;

    avctx->width = s->width;

    /* XXX: not complete test ! */

    switch((s->h_count[0] << 4) | s->v_count[0]) {

        case 0x11:

    	    avctx->pix_fmt = PIX_FMT_YUV444P;

            break;

        case 0x21:

            avctx->pix_fmt = PIX_FMT_YUV422P;

            break;

        default:

	case 0x22:

            avctx->pix_fmt = PIX_FMT_YUV420P;

            break;

    }

    /* dummy quality */

    /* XXX: infer it with matrix */

//    avctx->quality = 3; 



    return buf_ptr - buf;

}
