AVFilterBufferRef *avfilter_null_get_audio_buffer(AVFilterLink *link, int perms,

                                                  enum AVSampleFormat sample_fmt, int size,

                                                  int64_t channel_layout, int packed)

{

    return avfilter_get_audio_buffer(link->dst->outputs[0], perms, sample_fmt,

                                     size, channel_layout, packed);

}
