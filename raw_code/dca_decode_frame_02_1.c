static int dca_decode_frame(AVCodecContext * avctx,
                            void *data, int *data_size,
                            AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    int i;
    int16_t *samples = data;
    DCAContext *s = avctx->priv_data;
    int channels;
    s->dca_buffer_size = dca_convert_bitstream(buf, buf_size, s->dca_buffer, DCA_MAX_FRAME_SIZE);
    if (s->dca_buffer_size == -1) {
        av_log(avctx, AV_LOG_ERROR, "Not a valid DCA frame\n");
    }
    init_get_bits(&s->gb, s->dca_buffer, s->dca_buffer_size * 8);
    if (dca_parse_frame_header(s) < 0) {
        //seems like the frame is corrupt, try with the next one
        *data_size=0;
        return buf_size;
    }
    //set AVCodec values with parsed data
    avctx->sample_rate = s->sample_rate;
    avctx->bit_rate = s->bit_rate;
    channels = s->prim_channels + !!s->lfe;
    if (s->amode<16) {
        avctx->channel_layout = dca_core_channel_layout[s->amode];
        if (s->lfe) {
            avctx->channel_layout |= CH_LOW_FREQUENCY;
            s->channel_order_tab = dca_channel_reorder_lfe[s->amode];
        } else
            s->channel_order_tab = dca_channel_reorder_nolfe[s->amode];
        if(avctx->request_channels == 2 && s->prim_channels > 2) {
            channels = 2;
            s->output = DCA_STEREO;
            avctx->channel_layout = CH_LAYOUT_STEREO;
        }
    } else {
        av_log(avctx, AV_LOG_ERROR, "Non standard configuration %d !\n",s->amode);
    }
    /* There is nothing that prevents a dts frame to change channel configuration
       but FFmpeg doesn't support that so only set the channels if it is previously
       unset. Ideally during the first probe for channels the crc should be checked
       and only set avctx->channels when the crc is ok. Right now the decoder could
       set the channels based on a broken first frame.*/
    if (!avctx->channels)
        avctx->channels = channels;
    if(*data_size < (s->sample_blocks / 8) * 256 * sizeof(int16_t) * channels)
    *data_size = 256 / 8 * s->sample_blocks * sizeof(int16_t) * channels;
    for (i = 0; i < (s->sample_blocks / 8); i++) {
        dca_decode_block(s);
        s->dsp.float_to_int16_interleave(samples, s->samples_chanptr, 256, channels);
        samples += 256 * channels;
    }
    return buf_size;
}