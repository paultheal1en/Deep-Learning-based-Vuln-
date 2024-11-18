static int decode_tilehdr(WmallDecodeCtx *s)

{

    uint16_t num_samples[WMALL_MAX_CHANNELS] = { 0 }; /* sum of samples for all currently known subframes of a channel */

    uint8_t  contains_subframe[WMALL_MAX_CHANNELS];   /* flag indicating if a channel contains the current subframe */

    int channels_for_cur_subframe = s->num_channels;  /* number of channels that contain the current subframe */

    int fixed_channel_layout = 0;                     /* flag indicating that all channels use the same subfra2me offsets and sizes */

    int min_channel_len = 0;                          /* smallest sum of samples (channels with this length will be processed first) */

    int c, tile_aligned;



    /* reset tiling information */

    for (c = 0; c < s->num_channels; c++)

        s->channel[c].num_subframes = 0;



    tile_aligned = get_bits1(&s->gb);

    if (s->max_num_subframes == 1 || tile_aligned)

        fixed_channel_layout = 1;



    /* loop until the frame data is split between the subframes */

    do {

        int subframe_len, in_use = 0;



        /* check which channels contain the subframe */

        for (c = 0; c < s->num_channels; c++) {

            if (num_samples[c] == min_channel_len) {

                if (fixed_channel_layout || channels_for_cur_subframe == 1 ||

                   (min_channel_len == s->samples_per_frame - s->min_samples_per_subframe)) {

                    contains_subframe[c] = in_use = 1;

                } else {

                    if (get_bits1(&s->gb))

                        contains_subframe[c] = in_use = 1;

                }

            } else

                contains_subframe[c] = 0;

        }



        if (!in_use) {

            av_log(s->avctx, AV_LOG_ERROR,

                   "Found empty subframe\n");

            return AVERROR_INVALIDDATA;

        }



        /* get subframe length, subframe_len == 0 is not allowed */

        if ((subframe_len = decode_subframe_length(s, min_channel_len)) <= 0)

            return AVERROR_INVALIDDATA;

        /* add subframes to the individual channels and find new min_channel_len */

        min_channel_len += subframe_len;

        for (c = 0; c < s->num_channels; c++) {

            WmallChannelCtx *chan = &s->channel[c];



            if (contains_subframe[c]) {

                if (chan->num_subframes >= MAX_SUBFRAMES) {

                    av_log(s->avctx, AV_LOG_ERROR,

                           "broken frame: num subframes > 31\n");

                    return AVERROR_INVALIDDATA;

                }

                chan->subframe_len[chan->num_subframes] = subframe_len;

                num_samples[c] += subframe_len;

                ++chan->num_subframes;

                if (num_samples[c] > s->samples_per_frame) {

                    av_log(s->avctx, AV_LOG_ERROR, "broken frame: "

                           "channel len(%d) > samples_per_frame(%d)\n",

                           num_samples[c], s->samples_per_frame);

                    return AVERROR_INVALIDDATA;

                }

            } else if (num_samples[c] <= min_channel_len) {

                if (num_samples[c] < min_channel_len) {

                    channels_for_cur_subframe = 0;

                    min_channel_len = num_samples[c];

                }

                ++channels_for_cur_subframe;

            }

        }

    } while (min_channel_len < s->samples_per_frame);



    for (c = 0; c < s->num_channels; c++) {

        int i, offset = 0;

        for (i = 0; i < s->channel[c].num_subframes; i++) {

            s->channel[c].subframe_offsets[i] = offset;

            offset += s->channel[c].subframe_len[i];

        }

    }



    return 0;

}
