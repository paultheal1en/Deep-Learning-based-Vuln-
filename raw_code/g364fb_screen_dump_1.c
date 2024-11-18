static void g364fb_screen_dump(void *opaque, const char *filename)
{
    G364State *s = opaque;
    int y, x;
    uint8_t index;
    uint8_t *data_buffer;
    FILE *f;
    if (s->depth != 8) {
        error_report("g364: unknown guest depth %d", s->depth);
        return;
    }
    f = fopen(filename, "wb");
    if (!f)
        return;
    if (s->ctla & CTLA_FORCE_BLANK) {
        /* blank screen */
        fprintf(f, "P4\n%d %d\n",
            s->width, s->height);
        for (y = 0; y < s->height; y++)
            for (x = 0; x < s->width; x++)
                fputc(0, f);
    } else {
        data_buffer = s->vram + s->top_of_screen;
        fprintf(f, "P6\n%d %d\n%d\n",
            s->width, s->height, 255);
        for (y = 0; y < s->height; y++)
            for (x = 0; x < s->width; x++, data_buffer++) {
                index = *data_buffer;
                fputc(s->color_palette[index][0], f);
                fputc(s->color_palette[index][1], f);
                fputc(s->color_palette[index][2], f);
        }
    }
    fclose(f);
}