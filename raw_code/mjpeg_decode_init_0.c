static int mjpeg_decode_init(AVCodecContext *avctx)

{

    MJpegDecodeContext *s = avctx->priv_data;

    MpegEncContext s2;



    s->avctx = avctx;



    /* ugly way to get the idct & scantable */

    memset(&s2, 0, sizeof(MpegEncContext));

    s2.flags= avctx->flags;

    s2.avctx= avctx;

//    s2->out_format = FMT_MJPEG;

    s2.width = 8;

    s2.height = 8;

    if (MPV_common_init(&s2) < 0)

       return -1;

    s->scantable= s2.intra_scantable;

    s->idct_put= s2.idct_put;

    MPV_common_end(&s2);



    s->mpeg_enc_ctx_allocated = 0;

    s->buffer_size = 102400; /* smaller buffer should be enough,

				but photojpg files could ahive bigger sizes */

    s->buffer = av_malloc(s->buffer_size);

    if (!s->buffer)

	return -1;

    s->start_code = -1;

    s->first_picture = 1;

    s->org_width = avctx->width;

    s->org_height = avctx->height;

    

    build_vlc(&s->vlcs[0][0], bits_dc_luminance, val_dc_luminance, 12);

    build_vlc(&s->vlcs[0][1], bits_dc_chrominance, val_dc_chrominance, 12);

    build_vlc(&s->vlcs[1][0], bits_ac_luminance, val_ac_luminance, 251);

    build_vlc(&s->vlcs[1][1], bits_ac_chrominance, val_ac_chrominance, 251);



    if (avctx->flags & CODEC_FLAG_EXTERN_HUFF)

    {

	printf("mjpeg: using external huffman table\n");

	init_get_bits(&s->gb, avctx->extradata, avctx->extradata_size);

	mjpeg_decode_dht(s);

	/* should check for error - but dunno */

    }



    return 0;

}
