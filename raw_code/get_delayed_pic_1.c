static int get_delayed_pic(DiracContext *s, AVFrame *picture, int *got_frame)

{

    DiracFrame *out = s->delay_frames[0];

    int i, out_idx  = 0;

    int ret;



    /* find frame with lowest picture number */

    for (i = 1; s->delay_frames[i]; i++)

        if (s->delay_frames[i]->avframe->display_picture_number < out->avframe->display_picture_number) {

            out     = s->delay_frames[i];

            out_idx = i;

        }



    for (i = out_idx; s->delay_frames[i]; i++)

        s->delay_frames[i] = s->delay_frames[i+1];



    if (out) {

        out->reference ^= DELAYED_PIC_REF;

        *got_frame = 1;

        if((ret = av_frame_ref(picture, out->avframe)) < 0)

            return ret;

    }



    return 0;

}
