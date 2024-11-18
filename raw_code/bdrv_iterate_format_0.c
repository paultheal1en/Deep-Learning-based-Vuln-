void bdrv_iterate_format(void (*it)(void *opaque, const char *name),

                         void *opaque)

{

    BlockDriver *drv;

    int count = 0;

    int i;

    const char **formats = NULL;



    QLIST_FOREACH(drv, &bdrv_drivers, list) {

        if (drv->format_name) {

            bool found = false;

            int i = count;

            while (formats && i && !found) {

                found = !strcmp(formats[--i], drv->format_name);

            }



            if (!found) {

                formats = g_renew(const char *, formats, count + 1);

                formats[count++] = drv->format_name;

            }

        }

    }



    qsort(formats, count, sizeof(formats[0]), qsort_strcmp);



    for (i = 0; i < count; i++) {

        it(opaque, formats[i]);

    }



    g_free(formats);

}
