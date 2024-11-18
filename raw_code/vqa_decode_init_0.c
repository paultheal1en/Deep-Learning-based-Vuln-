static int vqa_decode_init(AVCodecContext *avctx)

{

    VqaContext *s = avctx->priv_data;

    unsigned char *vqa_header;

    int i, j, codebook_index;



    s->avctx = avctx;

    avctx->pix_fmt = PIX_FMT_PAL8;

    dsputil_init(&s->dsp, avctx);



    /* make sure the extradata made it */

    if (s->avctx->extradata_size != VQA_HEADER_SIZE) {

        av_log(s->avctx, AV_LOG_ERROR, "  VQA video: expected extradata size of %d\n", VQA_HEADER_SIZE);

        return -1;

    }



    /* load up the VQA parameters from the header */

    vqa_header = (unsigned char *)s->avctx->extradata;

    s->vqa_version = vqa_header[0];

    s->width = AV_RL16(&vqa_header[6]);

    s->height = AV_RL16(&vqa_header[8]);

    if(avcodec_check_dimensions(avctx, s->width, s->height)){

        s->width= s->height= 0;

        return -1;

    }

    s->vector_width = vqa_header[10];

    s->vector_height = vqa_header[11];

    s->partial_count = s->partial_countdown = vqa_header[13];



    /* the vector dimensions have to meet very stringent requirements */

    if ((s->vector_width != 4) ||

        ((s->vector_height != 2) && (s->vector_height != 4))) {

        /* return without further initialization */

        return -1;

    }



    /* allocate codebooks */

    s->codebook_size = MAX_CODEBOOK_SIZE;

    s->codebook = av_malloc(s->codebook_size);

    s->next_codebook_buffer = av_malloc(s->codebook_size);



    /* initialize the solid-color vectors */

    if (s->vector_height == 4) {

        codebook_index = 0xFF00 * 16;

        for (i = 0; i < 256; i++)

            for (j = 0; j < 16; j++)

                s->codebook[codebook_index++] = i;

    } else {

        codebook_index = 0xF00 * 8;

        for (i = 0; i < 256; i++)

            for (j = 0; j < 8; j++)

                s->codebook[codebook_index++] = i;

    }

    s->next_codebook_buffer_index = 0;



    /* allocate decode buffer */

    s->decode_buffer_size = (s->width / s->vector_width) *

        (s->height / s->vector_height) * 2;

    s->decode_buffer = av_malloc(s->decode_buffer_size);



    s->frame.data[0] = NULL;



    return 0;

}
