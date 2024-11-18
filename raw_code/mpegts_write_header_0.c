static int mpegts_write_header(AVFormatContext *s)

{

    MpegTSWrite *ts = s->priv_data;

    MpegTSWriteStream *ts_st;

    MpegTSService *service;

    AVStream *st, *pcr_st = NULL;

    AVDictionaryEntry *title, *provider;

    int i, j;

    const char *service_name;

    const char *provider_name;

    int *pids;

    int ret;



    if (s->max_delay < 0) /* Not set by the caller */

        s->max_delay = 0;



    // round up to a whole number of TS packets

    ts->pes_payload_size = (ts->pes_payload_size + 14 + 183) / 184 * 184 - 14;



    ts->tsid = ts->transport_stream_id;

    ts->onid = ts->original_network_id;

    /* allocate a single DVB service */

    title = av_dict_get(s->metadata, "service_name", NULL, 0);

    if (!title)

        title = av_dict_get(s->metadata, "title", NULL, 0);

    service_name = title ? title->value : DEFAULT_SERVICE_NAME;

    provider = av_dict_get(s->metadata, "service_provider", NULL, 0);

    provider_name = provider ? provider->value : DEFAULT_PROVIDER_NAME;

    service = mpegts_add_service(ts, ts->service_id, provider_name, service_name);

    service->pmt.write_packet = section_write_packet;

    service->pmt.opaque = s;

    service->pmt.cc = 15;



    ts->pat.pid = PAT_PID;

    ts->pat.cc = 15; // Initialize at 15 so that it wraps and be equal to 0 for the first packet we write

    ts->pat.write_packet = section_write_packet;

    ts->pat.opaque = s;



    ts->sdt.pid = SDT_PID;

    ts->sdt.cc = 15;

    ts->sdt.write_packet = section_write_packet;

    ts->sdt.opaque = s;



    pids = av_malloc(s->nb_streams * sizeof(*pids));

    if (!pids)

        return AVERROR(ENOMEM);



    /* assign pids to each stream */

    for(i = 0;i < s->nb_streams; i++) {

        st = s->streams[i];

        avpriv_set_pts_info(st, 33, 1, 90000);

        ts_st = av_mallocz(sizeof(MpegTSWriteStream));

        if (!ts_st) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        st->priv_data = ts_st;

        ts_st->payload = av_mallocz(ts->pes_payload_size);

        if (!ts_st->payload) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        ts_st->service = service;

        /* MPEG pid values < 16 are reserved. Applications which set st->id in

         * this range are assigned a calculated pid. */

        if (st->id < 16) {

            ts_st->pid = ts->start_pid + i;

        } else if (st->id < 0x1FFF) {

            ts_st->pid = st->id;

        } else {

            av_log(s, AV_LOG_ERROR, "Invalid stream id %d, must be less than 8191\n", st->id);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        if (ts_st->pid == service->pmt.pid) {

            av_log(s, AV_LOG_ERROR, "Duplicate stream id %d\n", ts_st->pid);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        for (j = 0; j < i; j++)

            if (pids[j] == ts_st->pid) {

                av_log(s, AV_LOG_ERROR, "Duplicate stream id %d\n", ts_st->pid);

                ret = AVERROR(EINVAL);

                goto fail;

            }

        pids[i] = ts_st->pid;

        ts_st->payload_pts = AV_NOPTS_VALUE;

        ts_st->payload_dts = AV_NOPTS_VALUE;

        ts_st->first_pts_check = 1;

        ts_st->cc = 15;

        /* update PCR pid by using the first video stream */

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

            service->pcr_pid == 0x1fff) {

            service->pcr_pid = ts_st->pid;

            pcr_st = st;

        }

        if (st->codec->codec_id == AV_CODEC_ID_AAC &&

            st->codec->extradata_size > 0)

        {

            AVStream *ast;

            ts_st->amux = avformat_alloc_context();

            if (!ts_st->amux) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            ts_st->amux->oformat = av_guess_format((ts->flags & MPEGTS_FLAG_AAC_LATM) ? "latm" : "adts", NULL, NULL);

            if (!ts_st->amux->oformat) {

                ret = AVERROR(EINVAL);

                goto fail;

            }

            ast = avformat_new_stream(ts_st->amux, NULL);

            ret = avcodec_copy_context(ast->codec, st->codec);

            if (ret != 0)

                goto fail;

            ret = avformat_write_header(ts_st->amux, NULL);

            if (ret < 0)

                goto fail;

        }

    }



    av_free(pids);



    /* if no video stream, use the first stream as PCR */

    if (service->pcr_pid == 0x1fff && s->nb_streams > 0) {

        pcr_st = s->streams[0];

        ts_st = pcr_st->priv_data;

        service->pcr_pid = ts_st->pid;

    }



    if (ts->mux_rate > 1) {

        service->pcr_packet_period = (ts->mux_rate * PCR_RETRANS_TIME) /

            (TS_PACKET_SIZE * 8 * 1000);

        ts->sdt_packet_period      = (ts->mux_rate * SDT_RETRANS_TIME) /

            (TS_PACKET_SIZE * 8 * 1000);

        ts->pat_packet_period      = (ts->mux_rate * PAT_RETRANS_TIME) /

            (TS_PACKET_SIZE * 8 * 1000);



        ts->first_pcr = av_rescale(s->max_delay, PCR_TIME_BASE, AV_TIME_BASE);

    } else {

        /* Arbitrary values, PAT/PMT could be written on key frames */

        ts->sdt_packet_period = 200;

        ts->pat_packet_period = 40;

        if (pcr_st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (!pcr_st->codec->frame_size) {

                av_log(s, AV_LOG_WARNING, "frame size not set\n");

                service->pcr_packet_period =

                    pcr_st->codec->sample_rate/(10*512);

            } else {

                service->pcr_packet_period =

                    pcr_st->codec->sample_rate/(10*pcr_st->codec->frame_size);

            }

        } else {

            // max delta PCR 0.1s

            service->pcr_packet_period =

                pcr_st->codec->time_base.den/(10*pcr_st->codec->time_base.num);

        }

    }



    // output a PCR as soon as possible

    service->pcr_packet_count = service->pcr_packet_period;

    ts->pat_packet_count = ts->pat_packet_period-1;

    ts->sdt_packet_count = ts->sdt_packet_period-1;



    if (ts->mux_rate == 1)

        av_log(s, AV_LOG_VERBOSE, "muxrate VBR, ");

    else

        av_log(s, AV_LOG_VERBOSE, "muxrate %d, ", ts->mux_rate);

    av_log(s, AV_LOG_VERBOSE, "pcr every %d pkts, "

           "sdt every %d, pat/pmt every %d pkts\n",

           service->pcr_packet_period,

           ts->sdt_packet_period, ts->pat_packet_period);



    avio_flush(s->pb);



    return 0;



 fail:

    av_free(pids);

    for(i = 0;i < s->nb_streams; i++) {

        MpegTSWriteStream *ts_st;

        st = s->streams[i];

        ts_st = st->priv_data;

        if (ts_st) {

            av_freep(&ts_st->payload);

            if (ts_st->amux) {

                avformat_free_context(ts_st->amux);

                ts_st->amux = NULL;

            }

        }

        av_freep(&st->priv_data);

    }

    return ret;

}
