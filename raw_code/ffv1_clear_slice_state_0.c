void ffv1_clear_slice_state(FFV1Context *f, FFV1Context *fs)

{

    int i, j;



    for (i = 0; i < f->plane_count; i++) {

        PlaneContext *p = &fs->plane[i];



        p->interlace_bit_state[0] = 128;

        p->interlace_bit_state[1] = 128;



        if (fs->ac) {

            if (f->initial_states[p->quant_table_index]) {

                memcpy(p->state, f->initial_states[p->quant_table_index],

                       CONTEXT_SIZE * p->context_count);

            } else

                memset(p->state, 128, CONTEXT_SIZE * p->context_count);

        } else {

            for (j = 0; j < p->context_count; j++) {

                p->vlc_state[j].drift     = 0;

                p->vlc_state[j].error_sum = 4;    //FFMAX((RANGE + 32)/64, 2);

                p->vlc_state[j].bias      = 0;

                p->vlc_state[j].count     = 1;

            }

        }

    }

}
