static void vfio_bar_quirk_teardown(VFIODevice *vdev, int nr)

{

    VFIOBAR *bar = &vdev->bars[nr];



    while (!QLIST_EMPTY(&bar->quirks)) {

        VFIOQuirk *quirk = QLIST_FIRST(&bar->quirks);

        memory_region_del_subregion(&bar->mem, &quirk->mem);


        QLIST_REMOVE(quirk, next);

        g_free(quirk);

    }

}