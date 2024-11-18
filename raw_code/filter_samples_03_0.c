static void filter_samples(AVFilterLink *inlink, AVFilterBufferRef *samplesref)

{

    AVFilterContext *ctx = inlink->dst;

    ShowInfoContext *showinfo = ctx->priv;

    uint32_t plane_checksum[8] = {0}, checksum = 0;

    char chlayout_str[128];

    int plane;



    for (plane = 0; samplesref->data[plane] && plane < 8; plane++) {

        uint8_t *data = samplesref->data[plane];

        int linesize = samplesref->linesize[plane];



        plane_checksum[plane] = av_adler32_update(plane_checksum[plane],

                                                  data, linesize);

        checksum = av_adler32_update(checksum, data, linesize);

    }



    av_get_channel_layout_string(chlayout_str, sizeof(chlayout_str), -1,

                                 samplesref->audio->channel_layout);



    av_log(ctx, AV_LOG_INFO,

           "n:%d pts:%"PRId64" pts_time:%f pos:%"PRId64" "

           "fmt:%s chlayout:%s nb_samples:%d rate:%d planar:%d "

           "checksum:%u plane_checksum[%u %u %u %u %u %u %u %u]\n",

           showinfo->frame,

           samplesref->pts, samplesref->pts * av_q2d(inlink->time_base),

           samplesref->pos,

           av_get_sample_fmt_name(samplesref->format),

           chlayout_str,

           samplesref->audio->nb_samples,

           samplesref->audio->sample_rate,

           samplesref->audio->planar,

           checksum,

           plane_checksum[0], plane_checksum[1], plane_checksum[2], plane_checksum[3],

           plane_checksum[4], plane_checksum[5], plane_checksum[6], plane_checksum[7]);



    showinfo->frame++;



    avfilter_filter_samples(inlink->dst->outputs[0], samplesref);

}
