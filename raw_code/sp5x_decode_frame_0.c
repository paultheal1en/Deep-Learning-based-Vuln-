static int sp5x_decode_frame(AVCodecContext *avctx, 

                              void *data, int *data_size,

                              uint8_t *buf, int buf_size)

{

#if 0

    MJpegDecodeContext *s = avctx->priv_data;

#endif

    const int qscale = 5;

    uint8_t *buf_ptr, *buf_end, *recoded;

    int i = 0, j = 0;



    /* no supplementary picture */

    if (buf_size == 0)

        return 0;



    if (!avctx->width || !avctx->height)

	return -1;



    buf_ptr = buf;

    buf_end = buf + buf_size;



#if 1

    recoded = av_mallocz(buf_size + 1024);

    if (!recoded)

	return -1;



    /* SOI */

    recoded[j++] = 0xFF;

    recoded[j++] = 0xD8;



    memcpy(recoded+j, &sp5x_data_dqt[0], sizeof(sp5x_data_dqt));

    memcpy(recoded+j+5, &sp5x_quant_table[qscale * 2], 64);

    memcpy(recoded+j+70, &sp5x_quant_table[(qscale * 2) + 1], 64);

    j += sizeof(sp5x_data_dqt);



    memcpy(recoded+j, &sp5x_data_dht[0], sizeof(sp5x_data_dht));

    j += sizeof(sp5x_data_dht);



    memcpy(recoded+j, &sp5x_data_sof[0], sizeof(sp5x_data_sof));

    recoded[j+5] = (avctx->coded_height >> 8) & 0xFF;

    recoded[j+6] = avctx->coded_height & 0xFF;

    recoded[j+7] = (avctx->coded_width >> 8) & 0xFF;

    recoded[j+8] = avctx->coded_width & 0xFF;

    j += sizeof(sp5x_data_sof);



    memcpy(recoded+j, &sp5x_data_sos[0], sizeof(sp5x_data_sos));

    j += sizeof(sp5x_data_sos);



    for (i = 14; i < buf_size && j < buf_size+1024-2; i++)

    {

	recoded[j++] = buf[i];

	if (buf[i] == 0xff)

	    recoded[j++] = 0;

    }



    /* EOI */

    recoded[j++] = 0xFF;

    recoded[j++] = 0xD9;



    i = mjpeg_decode_frame(avctx, data, data_size, recoded, j);



    av_free(recoded);



#else

    /* SOF */

    s->bits = 8;

    s->width  = avctx->coded_width;

    s->height = avctx->coded_height;

    s->nb_components = 3;

    s->component_id[0] = 0;

    s->h_count[0] = 2;

    s->v_count[0] = 2;

    s->quant_index[0] = 0;

    s->component_id[1] = 1;

    s->h_count[1] = 1;

    s->v_count[1] = 1;

    s->quant_index[1] = 1;

    s->component_id[2] = 2;

    s->h_count[2] = 1;

    s->v_count[2] = 1;

    s->quant_index[2] = 1;

    s->h_max = 2;

    s->v_max = 2;

    

    s->qscale_table = av_mallocz((s->width+15)/16);

    avctx->pix_fmt = s->cs_itu601 ? PIX_FMT_YUV420P : PIX_FMT_YUVJ420;

    s->interlaced = 0;

    

    s->picture.reference = 0;

    if (avctx->get_buffer(avctx, &s->picture) < 0)

    {

	fprintf(stderr, "get_buffer() failed\n");

	return -1;

    }



    s->picture.pict_type = I_TYPE;

    s->picture.key_frame = 1;



    for (i = 0; i < 3; i++)

	s->linesize[i] = s->picture.linesize[i] << s->interlaced;



    /* DQT */

    for (i = 0; i < 64; i++)

    {

	j = s->scantable.permutated[i];

	s->quant_matrixes[0][j] = sp5x_quant_table[(qscale * 2) + i];

    }

    s->qscale[0] = FFMAX(

	s->quant_matrixes[0][s->scantable.permutated[1]],

	s->quant_matrixes[0][s->scantable.permutated[8]]) >> 1;



    for (i = 0; i < 64; i++)

    {

	j = s->scantable.permutated[i];

	s->quant_matrixes[1][j] = sp5x_quant_table[(qscale * 2) + 1 + i];

    }

    s->qscale[1] = FFMAX(

	s->quant_matrixes[1][s->scantable.permutated[1]],

	s->quant_matrixes[1][s->scantable.permutated[8]]) >> 1;



    /* DHT */



    /* SOS */

    s->comp_index[0] = 0;

    s->nb_blocks[0] = s->h_count[0] * s->v_count[0];

    s->h_scount[0] = s->h_count[0];

    s->v_scount[0] = s->v_count[0];

    s->dc_index[0] = 0;

    s->ac_index[0] = 0;



    s->comp_index[1] = 1;

    s->nb_blocks[1] = s->h_count[1] * s->v_count[1];

    s->h_scount[1] = s->h_count[1];

    s->v_scount[1] = s->v_count[1];

    s->dc_index[1] = 1;

    s->ac_index[1] = 1;



    s->comp_index[2] = 2;

    s->nb_blocks[2] = s->h_count[2] * s->v_count[2];

    s->h_scount[2] = s->h_count[2];

    s->v_scount[2] = s->v_count[2];

    s->dc_index[2] = 1;

    s->ac_index[2] = 1;

    

    for (i = 0; i < 3; i++)

	s->last_dc[i] = 1024;



    s->mb_width = (s->width * s->h_max * 8 -1) / (s->h_max * 8);

    s->mb_height = (s->height * s->v_max * 8 -1) / (s->v_max * 8);



    init_get_bits(&s->gb, buf+14, (buf_size-14)*8);

    

    return mjpeg_decode_scan(s);

#endif



    return i;

}
