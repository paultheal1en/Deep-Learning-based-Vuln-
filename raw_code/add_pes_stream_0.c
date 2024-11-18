static PESContext *add_pes_stream(MpegTSContext *ts, int pid, int pcr_pid, int stream_type)

{

    MpegTSFilter *tss;

    PESContext *pes;



    /* if no pid found, then add a pid context */

    pes = av_mallocz(sizeof(PESContext));

    if (!pes)

        return 0;

    pes->ts = ts;

    pes->stream = ts->stream;

    pes->pid = pid;

    pes->pcr_pid = pcr_pid;

    pes->stream_type = stream_type;

    pes->state = MPEGTS_SKIP;

    pes->pts = AV_NOPTS_VALUE;

    pes->dts = AV_NOPTS_VALUE;

    tss = mpegts_open_pes_filter(ts, pid, mpegts_push_data, pes);

    if (!tss) {

        av_free(pes);

        return 0;

    }

    return pes;

}
