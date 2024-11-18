static void tcx24_screen_dump(void *opaque, const char *filename, bool cswitch,

                              Error **errp)

{

    TCXState *s = opaque;

    FILE *f;

    uint8_t *d, *d1, v;

    uint32_t *s24, *cptr, dval;

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

    s24 = s->vram24;

    cptr = s->cplane;

    for(y = 0; y < s->height; y++) {

        d = d1;

        for(x = 0; x < s->width; x++, d++, s24++) {

            if ((*cptr++ & 0xff000000) == 0x03000000) { // 24-bit direct

                dval = *s24 & 0x00ffffff;

                ret = fputc((dval >> 16) & 0xff, f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc((dval >> 8) & 0xff, f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc(dval & 0xff, f);

                if (ret == EOF) {

                    goto write_err;

                }

            } else {

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

            }

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
