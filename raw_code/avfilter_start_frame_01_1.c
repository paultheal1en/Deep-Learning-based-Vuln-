void avfilter_start_frame(AVFilterLink *link, AVFilterPicRef *picref)

{

    void (*start_frame)(AVFilterLink *, AVFilterPicRef *);

    AVFilterPad *dst = &link_dpad(link);



    if(!(start_frame = dst->start_frame))

        start_frame = avfilter_default_start_frame;



    /* prepare to copy the picture if it has insufficient permissions */

    if((dst->min_perms & picref->perms) != dst->min_perms ||

        dst->rej_perms & picref->perms) {

        /*

        av_log(link->dst, AV_LOG_INFO,

                "frame copy needed (have perms %x, need %x, reject %x)\n",

                picref->perms,

                link_dpad(link).min_perms, link_dpad(link).rej_perms);

        */



        link->cur_pic = avfilter_default_get_video_buffer(link, dst->min_perms);

        link->srcpic = picref;

        link->cur_pic->pts = link->srcpic->pts;


    }

    else

        link->cur_pic = picref;



    start_frame(link, link->cur_pic);

}