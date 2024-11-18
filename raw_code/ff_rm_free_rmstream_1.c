void ff_rm_free_rmstream (RMStream *rms)

{

    av_free(rms->videobuf);

    av_free(rms->audiobuf);

}
