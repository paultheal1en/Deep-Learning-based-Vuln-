static AVFilterBufferRef *get_video_buffer(AVFilterLink *link, int perms,

                                        int w, int h)

{

    FlipContext *flip = link->dst->priv;

    int i;



    AVFilterBufferRef *picref = avfilter_get_video_buffer(link->dst->outputs[0],

                                                       perms, w, h);



    for (i = 0; i < 4; i ++) {

        int vsub = i == 1 || i == 2 ? flip->vsub : 0;



        if (picref->data[i]) {

            picref->data[i] += ((h >> vsub)-1) * picref->linesize[i];

            picref->linesize[i] = -picref->linesize[i];

        }

    }



    return picref;

}
