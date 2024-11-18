static void save_display_set(DVBSubContext *ctx)
{
    DVBSubRegion *region;
    DVBSubRegionDisplay *display;
    DVBSubCLUT *clut;
    uint32_t *clut_table;
    int x_pos, y_pos, width, height;
    int x, y, y_off, x_off;
    uint32_t *pbuf;
    char filename[32];
    static int fileno_index = 0;
    x_pos = -1;
    y_pos = -1;
    width = 0;
    height = 0;
    for (display = ctx->display_list; display; display = display->next) {
        region = get_region(ctx, display->region_id);
        if (x_pos == -1) {
            x_pos = display->x_pos;
            y_pos = display->y_pos;
            width = region->width;
            height = region->height;
        } else {
            if (display->x_pos < x_pos) {
                width += (x_pos - display->x_pos);
                x_pos = display->x_pos;
            }
            if (display->y_pos < y_pos) {
                height += (y_pos - display->y_pos);
                y_pos = display->y_pos;
            }
            if (display->x_pos + region->width > x_pos + width) {
                width = display->x_pos + region->width - x_pos;
            }
            if (display->y_pos + region->height > y_pos + height) {
                height = display->y_pos + region->height - y_pos;
            }
        }
    }
    if (x_pos >= 0) {
        pbuf = av_malloc(width * height * 4);
        for (display = ctx->display_list; display; display = display->next) {
            region = get_region(ctx, display->region_id);
            x_off = display->x_pos - x_pos;
            y_off = display->y_pos - y_pos;
            clut = get_clut(ctx, region->clut);
            if (!clut)
                clut = &default_clut;
            switch (region->depth) {
            case 2:
                clut_table = clut->clut4;
                break;
            case 8:
                clut_table = clut->clut256;
                break;
            case 4:
            default:
                clut_table = clut->clut16;
                break;
            }
            for (y = 0; y < region->height; y++) {
                for (x = 0; x < region->width; x++) {
                    pbuf[((y + y_off) * width) + x_off + x] =
                        clut_table[region->pbuf[y * region->width + x]];
                }
            }
        }
        snprintf(filename, sizeof(filename), "dvbs.%d", fileno_index);
        png_save2(filename, pbuf, width, height);
        av_freep(&pbuf);
    }
    fileno_index++;
}