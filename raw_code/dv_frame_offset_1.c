static int64_t dv_frame_offset(AVFormatContext *s, DVDemuxContext *c,

                               int64_t timestamp, int flags)

{

    // FIXME: sys may be wrong if last dv_read_packet() failed (buffer is junk)

    const AVDVProfile *sys = av_dv_codec_profile2(c->vst->codec->width, c->vst->codec->height,

                                                 c->vst->codec->pix_fmt, c->vst->codec->time_base);

    int64_t offset;

    int64_t size       = avio_size(s->pb) - s->internal->data_offset;

    int64_t max_offset = ((size - 1) / sys->frame_size) * sys->frame_size;



    offset = sys->frame_size * timestamp;



    if (size >= 0 && offset > max_offset)

        offset = max_offset;

    else if (offset < 0)

        offset = 0;



    return offset + s->internal->data_offset;

}
