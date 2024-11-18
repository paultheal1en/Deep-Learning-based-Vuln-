static void vfio_vga_quirk_teardown(VFIODevice *vdev)

{

    int i;



    for (i = 0; i < ARRAY_SIZE(vdev->vga.region); i++) {

        while (!QLIST_EMPTY(&vdev->vga.region[i].quirks)) {

            VFIOQuirk *quirk = QLIST_FIRST(&vdev->vga.region[i].quirks);

            memory_region_del_subregion(&vdev->vga.region[i].mem, &quirk->mem);


            QLIST_REMOVE(quirk, next);

            g_free(quirk);

        }

    }

}