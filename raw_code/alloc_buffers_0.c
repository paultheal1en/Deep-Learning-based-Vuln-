static av_cold int alloc_buffers(AVCodecContext *avctx, AACEncContext *s)

{

    FF_ALLOCZ_OR_GOTO(avctx, s->buffer.samples, 3 * 1024 * s->channels * sizeof(s->buffer.samples[0]), alloc_fail);

    FF_ALLOCZ_OR_GOTO(avctx, s->cpe, sizeof(ChannelElement) * s->chan_map[0], alloc_fail);

    FF_ALLOCZ_OR_GOTO(avctx, avctx->extradata, 5 + FF_INPUT_BUFFER_PADDING_SIZE, alloc_fail);



    for(int ch = 0; ch < s->channels; ch++)

        s->planar_samples[ch] = s->buffer.samples + 3 * 1024 * ch;



    return 0;

alloc_fail:

    return AVERROR(ENOMEM);

}
