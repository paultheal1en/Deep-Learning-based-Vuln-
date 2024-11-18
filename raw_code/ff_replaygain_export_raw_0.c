int ff_replaygain_export_raw(AVStream *st, int32_t tg, uint32_t tp,

                             int32_t ag, uint32_t ap)

{

    AVReplayGain *replaygain;



    if (tg == INT32_MIN && ag == INT32_MIN)

        return 0;



    replaygain = (AVReplayGain*)ff_stream_new_side_data(st, AV_PKT_DATA_REPLAYGAIN,

                                                        sizeof(*replaygain));

    if (!replaygain)

        return AVERROR(ENOMEM);



    replaygain->track_gain = tg;

    replaygain->track_peak = tp;

    replaygain->album_gain = ag;

    replaygain->album_peak = ap;



    return 0;

}
