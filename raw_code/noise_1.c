static int noise(AVBitStreamFilterContext *bsfc, AVCodecContext *avctx, const char *args,

                     uint8_t **poutbuf, int *poutbuf_size,

                     const uint8_t *buf, int buf_size, int keyframe){

    unsigned int *state= bsfc->priv_data;

    int amount= args ? atoi(args) : (*state % 10001+1);

    int i;



    if(amount <= 0)

        return AVERROR(EINVAL);



    *poutbuf= av_malloc(buf_size + FF_INPUT_BUFFER_PADDING_SIZE);





    memcpy(*poutbuf, buf, buf_size + FF_INPUT_BUFFER_PADDING_SIZE);

    for(i=0; i<buf_size; i++){

        (*state) += (*poutbuf)[i] + 1;

        if(*state % amount == 0)

            (*poutbuf)[i] = *state;

    }

    return 1;

}