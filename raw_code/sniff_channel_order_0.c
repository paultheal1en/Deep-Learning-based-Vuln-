static uint64_t sniff_channel_order(uint8_t (*layout_map)[3], int tags)

{

    int i, n, total_non_cc_elements;

    struct elem_to_channel e2c_vec[MAX_ELEM_ID] = {{ 0 }};

    int num_front_channels, num_side_channels, num_back_channels;

    uint64_t layout;



    i = 0;

    num_front_channels =

        count_paired_channels(layout_map, tags, AAC_CHANNEL_FRONT, &i);

    if (num_front_channels < 0)

        return 0;

    num_side_channels =

        count_paired_channels(layout_map, tags, AAC_CHANNEL_SIDE, &i);

    if (num_side_channels < 0)

        return 0;

    num_back_channels =

        count_paired_channels(layout_map, tags, AAC_CHANNEL_BACK, &i);

    if (num_back_channels < 0)

        return 0;



    i = 0;

    if (num_front_channels & 1) {

        e2c_vec[i] = (struct elem_to_channel) {

            .av_position = AV_CH_FRONT_CENTER, .syn_ele = TYPE_SCE,

            .elem_id = layout_map[i][1], .aac_position = AAC_CHANNEL_FRONT };

        i++;

        num_front_channels--;

    }

    if (num_front_channels >= 4) {

        i += assign_pair(e2c_vec, layout_map, i, tags,

                         AV_CH_FRONT_LEFT_OF_CENTER,

                         AV_CH_FRONT_RIGHT_OF_CENTER,

                         AAC_CHANNEL_FRONT);

        num_front_channels -= 2;

    }

    if (num_front_channels >= 2) {

        i += assign_pair(e2c_vec, layout_map, i, tags,

                         AV_CH_FRONT_LEFT,

                         AV_CH_FRONT_RIGHT,

                         AAC_CHANNEL_FRONT);

        num_front_channels -= 2;

    }

    while (num_front_channels >= 2) {

        i += assign_pair(e2c_vec, layout_map, i, tags,

                         UINT64_MAX,

                         UINT64_MAX,

                         AAC_CHANNEL_FRONT);

        num_front_channels -= 2;

    }



    if (num_side_channels >= 2) {

        i += assign_pair(e2c_vec, layout_map, i, tags,

                         AV_CH_SIDE_LEFT,

                         AV_CH_SIDE_RIGHT,

                         AAC_CHANNEL_FRONT);

        num_side_channels -= 2;

    }

    while (num_side_channels >= 2) {

        i += assign_pair(e2c_vec, layout_map, i, tags,

                         UINT64_MAX,

                         UINT64_MAX,

                         AAC_CHANNEL_SIDE);

        num_side_channels -= 2;

    }



    while (num_back_channels >= 4) {

        i += assign_pair(e2c_vec, layout_map, i, tags,

                         UINT64_MAX,

                         UINT64_MAX,

                         AAC_CHANNEL_BACK);

        num_back_channels -= 2;

    }

    if (num_back_channels >= 2) {

        i += assign_pair(e2c_vec, layout_map, i, tags,

                         AV_CH_BACK_LEFT,

                         AV_CH_BACK_RIGHT,

                         AAC_CHANNEL_BACK);

        num_back_channels -= 2;

    }

    if (num_back_channels) {

        e2c_vec[i] = (struct elem_to_channel) {

          .av_position = AV_CH_BACK_CENTER, .syn_ele = TYPE_SCE,

          .elem_id = layout_map[i][1], .aac_position = AAC_CHANNEL_BACK };

        i++;

        num_back_channels--;

    }



    if (i < tags && layout_map[i][2] == AAC_CHANNEL_LFE) {

        e2c_vec[i] = (struct elem_to_channel) {

          .av_position = AV_CH_LOW_FREQUENCY, .syn_ele = TYPE_LFE,

          .elem_id = layout_map[i][1], .aac_position = AAC_CHANNEL_LFE };

        i++;

    }

    while (i < tags && layout_map[i][2] == AAC_CHANNEL_LFE) {

        e2c_vec[i] = (struct elem_to_channel) {

          .av_position = UINT64_MAX, .syn_ele = TYPE_LFE,

          .elem_id = layout_map[i][1], .aac_position = AAC_CHANNEL_LFE };

        i++;

    }



    // Must choose a stable sort

    total_non_cc_elements = n = i;

    do {

        int next_n = 0;

        for (i = 1; i < n; i++) {

            if (e2c_vec[i-1].av_position > e2c_vec[i].av_position) {

                FFSWAP(struct elem_to_channel, e2c_vec[i-1], e2c_vec[i]);

                next_n = i;

            }

        }

        n = next_n;

    } while (n > 0);



    layout = 0;

    for (i = 0; i < total_non_cc_elements; i++) {

        layout_map[i][0] = e2c_vec[i].syn_ele;

        layout_map[i][1] = e2c_vec[i].elem_id;

        layout_map[i][2] = e2c_vec[i].aac_position;

        if (e2c_vec[i].av_position != UINT64_MAX) {

            layout |= e2c_vec[i].av_position;

        }

    }



    return layout;

}
