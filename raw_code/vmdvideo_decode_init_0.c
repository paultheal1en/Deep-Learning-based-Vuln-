static int vmdvideo_decode_init(AVCodecContext *avctx)

{

    VmdVideoContext *s = avctx->priv_data;

    int i;

    unsigned int *palette32;

    int palette_index = 0;

    unsigned char r, g, b;

    unsigned char *vmd_header;

    unsigned char *raw_palette;



    s->avctx = avctx;

    avctx->pix_fmt = PIX_FMT_PAL8;

    dsputil_init(&s->dsp, avctx);



    /* make sure the VMD header made it */

    if (s->avctx->extradata_size != VMD_HEADER_SIZE) {

        av_log(s->avctx, AV_LOG_ERROR, "VMD video: expected extradata size of %d\n",

            VMD_HEADER_SIZE);

        return -1;

    }

    vmd_header = (unsigned char *)avctx->extradata;



    s->unpack_buffer_size = AV_RL32(&vmd_header[800]);

    s->unpack_buffer = av_malloc(s->unpack_buffer_size);

    if (!s->unpack_buffer)

        return -1;



    /* load up the initial palette */

    raw_palette = &vmd_header[28];

    palette32 = (unsigned int *)s->palette;

    for (i = 0; i < PALETTE_COUNT; i++) {

        r = raw_palette[palette_index++] * 4;

        g = raw_palette[palette_index++] * 4;

        b = raw_palette[palette_index++] * 4;

        palette32[i] = (r << 16) | (g << 8) | (b);

    }



    s->frame.data[0] = s->prev_frame.data[0] = NULL;



    return 0;

}
