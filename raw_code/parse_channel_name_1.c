static int parse_channel_name(char **arg, int *rchannel, int *rnamed)

{

    char buf[8];

    int len, i, channel_id = 0;

    int64_t layout, layout0;



    /* try to parse a channel name, e.g. "FL" */

    if (sscanf(*arg, " %7[A-Z] %n", buf, &len)) {

        layout0 = layout = av_get_channel_layout(buf);

        /* channel_id <- first set bit in layout */

        for (i = 32; i > 0; i >>= 1) {

            if (layout >= (int64_t)1 << i) {

                channel_id += i;

                layout >>= i;

            }

        }

        /* reject layouts that are not a single channel */

        if (channel_id >= MAX_CHANNELS || layout0 != (int64_t)1 << channel_id)

            return AVERROR(EINVAL);

        *rchannel = channel_id;

        *rnamed = 1;

        *arg += len;

        return 0;

    }

    /* try to parse a channel number, e.g. "c2" */

    if (sscanf(*arg, " c%d %n", &channel_id, &len) &&

        channel_id >= 0 && channel_id < MAX_CHANNELS) {

        *rchannel = channel_id;

        *rnamed = 0;

        *arg += len;

        return 0;

    }

    return AVERROR(EINVAL);

}
