av_cold int ff_ac3_encode_init(AVCodecContext *avctx)

{

    AC3EncodeContext *s = avctx->priv_data;

    int ret, frame_size_58;



    s->avctx = avctx;



    s->eac3 = avctx->codec_id == AV_CODEC_ID_EAC3;



    ff_ac3_common_init();



    ret = validate_options(s);

    if (ret)

        return ret;



    avctx->frame_size = AC3_BLOCK_SIZE * s->num_blocks;

    avctx->delay      = AC3_BLOCK_SIZE;



    s->bitstream_mode = avctx->audio_service_type;

    if (s->bitstream_mode == AV_AUDIO_SERVICE_TYPE_KARAOKE)

        s->bitstream_mode = 0x7;



    s->bits_written    = 0;

    s->samples_written = 0;



    /* calculate crc_inv for both possible frame sizes */

    frame_size_58 = (( s->frame_size    >> 2) + ( s->frame_size    >> 4)) << 1;

    s->crc_inv[0] = pow_poly((CRC16_POLY >> 1), (8 * frame_size_58) - 16, CRC16_POLY);

    if (s->bit_alloc.sr_code == 1) {

        frame_size_58 = (((s->frame_size+2) >> 2) + ((s->frame_size+2) >> 4)) << 1;

        s->crc_inv[1] = pow_poly((CRC16_POLY >> 1), (8 * frame_size_58) - 16, CRC16_POLY);

    }



    /* set function pointers */

    if (CONFIG_AC3_FIXED_ENCODER && s->fixed_point) {

        s->mdct_end                     = ff_ac3_fixed_mdct_end;

        s->mdct_init                    = ff_ac3_fixed_mdct_init;

        s->allocate_sample_buffers      = ff_ac3_fixed_allocate_sample_buffers;

    } else if (CONFIG_AC3_ENCODER || CONFIG_EAC3_ENCODER) {

        s->mdct_end                     = ff_ac3_float_mdct_end;

        s->mdct_init                    = ff_ac3_float_mdct_init;

        s->allocate_sample_buffers      = ff_ac3_float_allocate_sample_buffers;

    }

    if (CONFIG_EAC3_ENCODER && s->eac3)

        s->output_frame_header = ff_eac3_output_frame_header;

    else

        s->output_frame_header = ac3_output_frame_header;



    set_bandwidth(s);



    exponent_init(s);



    bit_alloc_init(s);



    ret = s->mdct_init(s);

    if (ret)

        goto init_fail;



    ret = allocate_buffers(s);

    if (ret)

        goto init_fail;



    ff_audiodsp_init(&s->adsp);

    ff_me_cmp_init(&s->mecc, avctx);

    ff_ac3dsp_init(&s->ac3dsp, avctx->flags & CODEC_FLAG_BITEXACT);



    dprint_options(s);



    return 0;

init_fail:

    ff_ac3_encode_close(avctx);

    return ret;

}
