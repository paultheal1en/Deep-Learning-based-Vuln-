static void draw_slice(AVFilterLink *link, int y, int h)

{

    ScaleContext *scale = link->dst->priv;

    int out_h;

    AVFilterPicRef *cur_pic = link->cur_pic;

    uint8_t *data[4];



    if (!scale->slice_dir) {

        if (y != 0 && y + h != link->h) {

            av_log(scale, AV_LOG_ERROR, "Slices start in the middle!\n");

            return;

        }

        scale->slice_dir = y ?                       -1 : 1;

        scale->slice_y   = y ? link->dst->outputs[0]->h : y;

    }



    data[0] = cur_pic->data[0] +  y               * cur_pic->linesize[0];

    data[1] = scale->input_is_pal ?

              cur_pic->data[1] :

              cur_pic->data[1] + (y>>scale->vsub) * cur_pic->linesize[1];

    data[2] = cur_pic->data[2] + (y>>scale->vsub) * cur_pic->linesize[2];

    data[3] = cur_pic->data[3] +  y               * cur_pic->linesize[3];



    out_h = sws_scale(scale->sws, data, cur_pic->linesize, y, h,

                      link->dst->outputs[0]->outpic->data,

                      link->dst->outputs[0]->outpic->linesize);



    if (scale->slice_dir == -1)

        scale->slice_y -= out_h;

    avfilter_draw_slice(link->dst->outputs[0], scale->slice_y, out_h);

    if (scale->slice_dir == 1)

        scale->slice_y += out_h;

}
