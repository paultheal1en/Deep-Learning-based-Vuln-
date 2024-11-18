void av_opt_freep_ranges(AVOptionRanges **rangesp)

{

    int i;

    AVOptionRanges *ranges = *rangesp;



    if (!ranges)

        return;



    for (i = 0; i < ranges->nb_ranges * ranges->nb_components; i++) {

        AVOptionRange *range = ranges->range[i];

        av_freep(&range->str);

        av_freep(&ranges->range[i]);

    }

    av_freep(&ranges->range);

    av_freep(rangesp);

}
