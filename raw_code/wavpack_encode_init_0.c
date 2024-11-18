static av_cold int wavpack_encode_init(AVCodecContext *avctx)

{

    WavPackEncodeContext *s = avctx->priv_data;



    s->avctx = avctx;



    if (!avctx->frame_size) {

        int block_samples;

        if (!(avctx->sample_rate & 1))

            block_samples = avctx->sample_rate / 2;

        else

            block_samples = avctx->sample_rate;



        while (block_samples * avctx->channels > 150000)

            block_samples /= 2;



        while (block_samples * avctx->channels < 40000)

            block_samples *= 2;

        avctx->frame_size = block_samples;

    } else if (avctx->frame_size && (avctx->frame_size < 128 ||

                              avctx->frame_size > WV_MAX_SAMPLES)) {

        av_log(avctx, AV_LOG_ERROR, "invalid block size: %d\n", avctx->frame_size);

        return AVERROR(EINVAL);

    }



    if (avctx->compression_level != FF_COMPRESSION_DEFAULT) {

        if (avctx->compression_level >= 3) {

            s->decorr_filter = 3;

            s->num_passes = 9;

            if      (avctx->compression_level >= 8) {

                s->num_branches = 4;

                s->extra_flags = EXTRA_TRY_DELTAS|EXTRA_ADJUST_DELTAS|EXTRA_SORT_FIRST|EXTRA_SORT_LAST|EXTRA_BRANCHES;

            } else if (avctx->compression_level >= 7) {

                s->num_branches = 3;

                s->extra_flags = EXTRA_TRY_DELTAS|EXTRA_ADJUST_DELTAS|EXTRA_SORT_FIRST|EXTRA_BRANCHES;

            } else if (avctx->compression_level >= 6) {

                s->num_branches = 2;

                s->extra_flags = EXTRA_TRY_DELTAS|EXTRA_ADJUST_DELTAS|EXTRA_SORT_FIRST|EXTRA_BRANCHES;

            } else if (avctx->compression_level >= 5) {

                s->num_branches = 1;

                s->extra_flags = EXTRA_TRY_DELTAS|EXTRA_ADJUST_DELTAS|EXTRA_SORT_FIRST|EXTRA_BRANCHES;

            } else if (avctx->compression_level >= 4) {

                s->num_branches = 1;

                s->extra_flags = EXTRA_TRY_DELTAS|EXTRA_ADJUST_DELTAS|EXTRA_BRANCHES;

            }

        } else if (avctx->compression_level == 2) {

            s->decorr_filter = 2;

            s->num_passes = 4;

        } else if (avctx->compression_level == 1) {

            s->decorr_filter = 1;

            s->num_passes = 2;

        } else if (avctx->compression_level < 1) {

            s->decorr_filter = 0;

            s->num_passes = 0;

        }

    }



    s->num_decorrs = decorr_filter_sizes[s->decorr_filter];

    s->decorr_specs = decorr_filters[s->decorr_filter];



    s->delta_decay = 2.0;



    return 0;

}
