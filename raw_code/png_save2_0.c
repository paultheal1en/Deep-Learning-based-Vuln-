static void png_save2(const char *filename, uint32_t *bitmap, int w, int h)

{

    int x, y, v;

    FILE *f;

    char fname[40], fname2[40];

    char command[1024];



    snprintf(fname, 40, "%s.ppm", filename);



    f = fopen(fname, "w");

    if (!f) {

        perror(fname);

        exit(1);

    }

    fprintf(f, "P6\n"

            "%d %d\n"

            "%d\n",

            w, h, 255);

    for(y = 0; y < h; y++) {

        for(x = 0; x < w; x++) {

            v = bitmap[y * w + x];

            putc((v >> 16) & 0xff, f);

            putc((v >> 8) & 0xff, f);

            putc((v >> 0) & 0xff, f);

        }

    }

    fclose(f);





    snprintf(fname2, 40, "%s-a.pgm", filename);



    f = fopen(fname2, "w");

    if (!f) {

        perror(fname2);

        exit(1);

    }

    fprintf(f, "P5\n"

            "%d %d\n"

            "%d\n",

            w, h, 255);

    for(y = 0; y < h; y++) {

        for(x = 0; x < w; x++) {

            v = bitmap[y * w + x];

            putc((v >> 24) & 0xff, f);

        }

    }

    fclose(f);



    snprintf(command, 1024, "pnmtopng -alpha %s %s > %s.png 2> /dev/null", fname2, fname, filename);

    system(command);



    snprintf(command, 1024, "rm %s %s", fname, fname2);

    system(command);

}
