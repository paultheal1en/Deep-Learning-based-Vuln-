static int mjpeg_decode_init(AVCodecContext *avctx)

{

    MJpegDecodeContext *s = avctx->priv_data;

    MpegEncContext s2;



    s->avctx = avctx;



    /* ugly way to get the idct & scantable FIXME */

    memset(&s2, 0, sizeof(MpegEncContext));

    s2.avctx= avctx;

//    s2->out_format = FMT_MJPEG;

    dsputil_init(&s2.dsp, avctx);

    DCT_common_init(&s2);



    s->scantable= s2.intra_scantable;

    s->idct_put= s2.dsp.idct_put;



    s->mpeg_enc_ctx_allocated = 0;

    s->buffer_size = 102400; /* smaller buffer should be enough,

				but photojpg files could ahive bigger sizes */

    s->buffer = av_malloc(s->buffer_size);

    if (!s->buffer)

	return -1;

    s->start_code = -1;

    s->first_picture = 1;

    s->org_height = avctx->coded_height;

    

    build_vlc(&s->vlcs[0][0], bits_dc_luminance, val_dc_luminance, 12);

    build_vlc(&s->vlcs[0][1], bits_dc_chrominance, val_dc_chrominance, 12);

    build_vlc(&s->vlcs[1][0], bits_ac_luminance, val_ac_luminance, 251);

    build_vlc(&s->vlcs[1][1], bits_ac_chrominance, val_ac_chrominance, 251);



    if (avctx->flags & CODEC_FLAG_EXTERN_HUFF)

    {

	av_log(avctx, AV_LOG_INFO, "mjpeg: using external huffman table\n");

	init_get_bits(&s->gb, avctx->extradata, avctx->extradata_size*8);

	mjpeg_decode_dht(s);

	/* should check for error - but dunno */

    }



    return 0;

}
