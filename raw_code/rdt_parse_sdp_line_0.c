rdt_parse_sdp_line (AVFormatContext *s, int st_index,

                    PayloadContext *rdt, const char *line)

{

    AVStream *stream = s->streams[st_index];

    const char *p = line;



    if (av_strstart(p, "OpaqueData:buffer;", &p)) {

        rdt->mlti_data = rdt_parse_b64buf(&rdt->mlti_data_size, p);

    } else if (av_strstart(p, "StartTime:integer;", &p))

        stream->first_dts = atoi(p);

    else if (av_strstart(p, "ASMRuleBook:string;", &p)) {

        int n, first = -1;



        for (n = 0; n < s->nb_streams; n++)

            if (s->streams[n]->id == stream->id) {

                int count = s->streams[n]->index + 1;

                if (first == -1) first = n;

                if (rdt->nb_rmst < count) {

                    RMStream **rmst= av_realloc(rdt->rmst, count*sizeof(*rmst));

                    if (!rmst)

                        return AVERROR(ENOMEM);

                    memset(rmst + rdt->nb_rmst, 0,

                           (count - rdt->nb_rmst) * sizeof(*rmst));

                    rdt->rmst    = rmst;

                    rdt->nb_rmst = count;

                }

                rdt->rmst[s->streams[n]->index] = ff_rm_alloc_rmstream();

                rdt_load_mdpr(rdt, s->streams[n], (n - first) * 2);



                if (s->streams[n]->codec->codec_id == CODEC_ID_AAC)

                    s->streams[n]->codec->frame_size = 1; // FIXME

           }

    }



    return 0;

}
