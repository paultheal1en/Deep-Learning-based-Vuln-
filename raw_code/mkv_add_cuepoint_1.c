static int mkv_add_cuepoint(mkv_cues *cues, int stream, int64_t ts, int64_t cluster_pos)

{

    mkv_cuepoint *entries = cues->entries;



    entries = av_realloc(entries, (cues->num_entries + 1) * sizeof(mkv_cuepoint));

    if (entries == NULL)

        return AVERROR(ENOMEM);



    if (ts < 0)

        return 0;



    entries[cues->num_entries  ].pts = ts;

    entries[cues->num_entries  ].tracknum = stream + 1;

    entries[cues->num_entries++].cluster_pos = cluster_pos - cues->segment_offset;



    cues->entries = entries;

    return 0;

}
