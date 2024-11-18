static int xvid_ff_2pass_destroy(struct xvid_context *ref,

                                xvid_plg_destroy_t *param) {

    /* Currently cannot think of anything to do on destruction */

    /* Still, the framework should be here for reference/use */

    if( ref->twopassbuffer != NULL )

        ref->twopassbuffer[0] = 0;

    return 0;

}
