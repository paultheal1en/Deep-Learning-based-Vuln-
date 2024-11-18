static void swap_channel_layouts_on_filter(AVFilterContext *filter)

{

    AVFilterLink *link = NULL;

    int i, j, k;



    for (i = 0; i < filter->nb_inputs; i++) {

        link = filter->inputs[i];



        if (link->type == AVMEDIA_TYPE_AUDIO &&

            link->out_channel_layouts->nb_channel_layouts == 1)

            break;

    }

    if (i == filter->nb_inputs)

        return;



    for (i = 0; i < filter->nb_outputs; i++) {

        AVFilterLink *outlink = filter->outputs[i];

        int best_idx, best_score = INT_MIN, best_count_diff = INT_MAX;



        if (outlink->type != AVMEDIA_TYPE_AUDIO ||

            outlink->in_channel_layouts->nb_channel_layouts < 2)

            continue;



        for (j = 0; j < outlink->in_channel_layouts->nb_channel_layouts; j++) {

            uint64_t  in_chlayout = link->out_channel_layouts->channel_layouts[0];

            uint64_t out_chlayout = outlink->in_channel_layouts->channel_layouts[j];

            int  in_channels      = av_get_channel_layout_nb_channels(in_chlayout);

            int out_channels      = av_get_channel_layout_nb_channels(out_chlayout);

            int count_diff        = out_channels - in_channels;

            int matched_channels, extra_channels;

            int score = 0;



            /* channel substitution */

            for (k = 0; k < FF_ARRAY_ELEMS(ch_subst); k++) {

                uint64_t cmp0 = ch_subst[k][0];

                uint64_t cmp1 = ch_subst[k][1];

                if (( in_chlayout & cmp0) && (!(out_chlayout & cmp0)) &&

                    (out_chlayout & cmp1) && (!( in_chlayout & cmp1))) {

                    in_chlayout  &= ~cmp0;

                    out_chlayout &= ~cmp1;

                    /* add score for channel match, minus a deduction for

                       having to do the substitution */

                    score += 10 * av_get_channel_layout_nb_channels(cmp1) - 2;

                }

            }



            /* no penalty for LFE channel mismatch */

            if ( (in_chlayout & AV_CH_LOW_FREQUENCY) &&

                (out_chlayout & AV_CH_LOW_FREQUENCY))

                score += 10;

            in_chlayout  &= ~AV_CH_LOW_FREQUENCY;

            out_chlayout &= ~AV_CH_LOW_FREQUENCY;



            matched_channels = av_get_channel_layout_nb_channels(in_chlayout &

                                                                 out_chlayout);

            extra_channels   = av_get_channel_layout_nb_channels(out_chlayout &

                                                                 (~in_chlayout));

            score += 10 * matched_channels - 5 * extra_channels;



            if (score > best_score ||

                (count_diff < best_count_diff && score == best_score)) {

                best_score = score;

                best_idx   = j;

                best_count_diff = count_diff;

            }

        }

        FFSWAP(uint64_t, outlink->in_channel_layouts->channel_layouts[0],

               outlink->in_channel_layouts->channel_layouts[best_idx]);

    }



}
