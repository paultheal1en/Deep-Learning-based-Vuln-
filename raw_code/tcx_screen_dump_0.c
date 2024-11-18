static void tcx_screen_dump(void *opaque, const char *filename, bool cswitch,

                            Error **errp)

{

    TCXState *s = opaque;

    FILE *f;

    uint8_t *d, *d1, v;

    int ret, y, x;



    f = fopen(filename, "wb");

    if (!f) {

        error_setg(errp, "failed to open file '%s': %s", filename,

                   strerror(errno));

        return;

    }

    ret = fprintf(f, "P6\n%d %d\n%d\n", s->width, s->height, 255);

    if (ret < 0) {

        goto write_err;

    }

    d1 = s->vram;

    for(y = 0; y < s->height; y++) {

        d = d1;

        for(x = 0; x < s->width; x++) {

            v = *d;

            ret = fputc(s->r[v], f);

            if (ret == EOF) {

                goto write_err;

            }

            ret = fputc(s->g[v], f);

            if (ret == EOF) {

                goto write_err;

            }

            ret = fputc(s->b[v], f);

            if (ret == EOF) {

                goto write_err;

            }

            d++;

        }

        d1 += MAXX;

    }



out:

    fclose(f);

    return;



write_err:

    error_setg(errp, "failed to write to file '%s': %s", filename,

               strerror(errno));

    unlink(filename);

    goto out;

}
