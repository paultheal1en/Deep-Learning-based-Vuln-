void ppm_save(const char *filename, struct DisplaySurface *ds, Error **errp)

{

    int width = pixman_image_get_width(ds->image);

    int height = pixman_image_get_height(ds->image);

    FILE *f;

    int y;

    int ret;

    pixman_image_t *linebuf;



    trace_ppm_save(filename, ds);

    f = fopen(filename, "wb");

    if (!f) {

        error_setg(errp, "failed to open file '%s': %s", filename,

                   strerror(errno));

        return;

    }

    ret = fprintf(f, "P6\n%d %d\n%d\n", width, height, 255);

    if (ret < 0) {

        linebuf = NULL;

        goto write_err;

    }

    linebuf = qemu_pixman_linebuf_create(PIXMAN_BE_r8g8b8, width);

    for (y = 0; y < height; y++) {

        qemu_pixman_linebuf_fill(linebuf, ds->image, width, 0, y);

        clearerr(f);

        ret = fwrite(pixman_image_get_data(linebuf), 1,

                     pixman_image_get_stride(linebuf), f);

        (void)ret;

        if (ferror(f)) {

            goto write_err;

        }

    }



out:

    qemu_pixman_image_unref(linebuf);

    fclose(f);

    return;



write_err:

    error_setg(errp, "failed to write to file '%s': %s", filename,

               strerror(errno));

    unlink(filename);

    goto out;

}
