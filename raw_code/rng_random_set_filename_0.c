static void rng_random_set_filename(Object *obj, const char *filename,

                                 Error **errp)

{

    RngBackend *b = RNG_BACKEND(obj);

    RndRandom *s = RNG_RANDOM(obj);



    if (b->opened) {

        error_set(errp, QERR_PERMISSION_DENIED);

        return;

    }



    if (s->filename) {

        g_free(s->filename);

    }



    s->filename = g_strdup(filename);

}
