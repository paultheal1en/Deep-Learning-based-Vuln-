static int ivi_init_tiles(IVIBandDesc *band, IVITile *ref_tile,

                          int p, int b, int t_height, int t_width)

{

    int x, y;

    IVITile *tile = band->tiles;



    for (y = 0; y < band->height; y += t_height) {

        for (x = 0; x < band->width; x += t_width) {

            tile->xpos     = x;

            tile->ypos     = y;

            tile->mb_size  = band->mb_size;

            tile->width    = FFMIN(band->width - x,  t_width);

            tile->height   = FFMIN(band->height - y, t_height);

            tile->is_empty = tile->data_size = 0;

            /* calculate number of macroblocks */

            tile->num_MBs  = IVI_MBs_PER_TILE(tile->width, tile->height,

                                              band->mb_size);



            av_freep(&tile->mbs);

            tile->mbs = av_malloc(tile->num_MBs * sizeof(IVIMbInfo));

            if (!tile->mbs)

                return AVERROR(ENOMEM);



            tile->ref_mbs = 0;

            if (p || b) {

                if (tile->num_MBs != ref_tile->num_MBs) {

                    av_log(NULL, AV_LOG_DEBUG, "ref_tile mismatch\n");

                    return AVERROR_INVALIDDATA;

                }

                tile->ref_mbs = ref_tile->mbs;

                ref_tile++;

            }

            tile++;

        }

    }



    return 0;

}
