int ffv1_init_slice_state(FFV1Context *f, FFV1Context *fs)

{

    int j;



    fs->plane_count  = f->plane_count;

    fs->transparency = f->transparency;

    for (j = 0; j < f->plane_count; j++) {

        PlaneContext *const p = &fs->plane[j];



        if (fs->ac) {

            if (!p->state)

                p->state = av_malloc(CONTEXT_SIZE * p->context_count *

                                     sizeof(uint8_t));

            if (!p->state)

                return AVERROR(ENOMEM);

        } else {

            if (!p->vlc_state)

                p->vlc_state = av_malloc(p->context_count * sizeof(VlcState));

            if (!p->vlc_state)

                return AVERROR(ENOMEM);

        }

    }



    if (fs->ac > 1) {

        //FIXME only redo if state_transition changed

        for (j = 1; j < 256; j++) {

            fs->c.one_state[j]        = f->state_transition[j];

            fs->c.zero_state[256 - j] = 256 - fs->c.one_state[j];

        }

    }



    return 0;

}
