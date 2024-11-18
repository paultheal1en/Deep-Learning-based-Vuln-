static void g364fb_screen_dump(void *opaque, const char *filename, bool cswitch,

                               Error **errp)

{

    G364State *s = opaque;

    int ret, y, x;

    uint8_t index;

    uint8_t *data_buffer;

    FILE *f;



    qemu_flush_coalesced_mmio_buffer();



    if (s->depth != 8) {

        error_setg(errp, "g364: unknown guest depth %d", s->depth);

        return;

    }



    f = fopen(filename, "wb");

    if (!f) {

        error_setg(errp, "failed to open file '%s': %s", filename,

                   strerror(errno));

        return;

    }



    if (s->ctla & CTLA_FORCE_BLANK) {

        /* blank screen */

        ret = fprintf(f, "P4\n%d %d\n", s->width, s->height);

        if (ret < 0) {

            goto write_err;

        }

        for (y = 0; y < s->height; y++)

            for (x = 0; x < s->width; x++) {

                ret = fputc(0, f);

                if (ret == EOF) {

                    goto write_err;

                }

            }

    } else {

        data_buffer = s->vram + s->top_of_screen;

        ret = fprintf(f, "P6\n%d %d\n%d\n", s->width, s->height, 255);

        if (ret < 0) {

            goto write_err;

        }

        for (y = 0; y < s->height; y++)

            for (x = 0; x < s->width; x++, data_buffer++) {

                index = *data_buffer;

                ret = fputc(s->color_palette[index][0], f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc(s->color_palette[index][1], f);

                if (ret == EOF) {

                    goto write_err;

                }

                ret = fputc(s->color_palette[index][2], f);

                if (ret == EOF) {

                    goto write_err;

                }

        }

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
