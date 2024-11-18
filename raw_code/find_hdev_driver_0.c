static BlockDriver *find_hdev_driver(const char *filename)

{

    int score_max = 0, score;

    BlockDriver *drv = NULL, *d;



    QLIST_FOREACH(d, &bdrv_drivers, list) {

        if (d->bdrv_probe_device) {

            score = d->bdrv_probe_device(filename);

            if (score > score_max) {

                score_max = score;

                drv = d;

            }

        }

    }



    return drv;

}
